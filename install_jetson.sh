#!/bin/bash
# Jetson Orin Nano一键安装脚本
# 自动安装和配置SITL项目环境

set -e  # 遇到错误立即退出

echo "🚀 Jetson Orin Nano SITL项目一键安装脚本"
echo "================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为Jetson设备
check_jetson() {
    log_info "检查Jetson设备..."
    if [ -f /etc/nv_tegra_release ]; then
        JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown")
        log_success "检测到Jetson设备: $JETSON_MODEL"
    else
        log_error "未检测到Jetson设备"
        exit 1
    fi
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查内存
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 6 ]; then
        log_warning "内存不足8GB，可能影响性能"
    else
        log_success "内存检查通过: ${MEMORY_GB}GB"
    fi
    
    # 检查存储空间
    STORAGE_GB=$(df -BG / | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$STORAGE_GB" -lt 10 ]; then
        log_error "存储空间不足，至少需要10GB"
        exit 1
    else
        log_success "存储空间检查通过: ${STORAGE_GB}GB可用"
    fi
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_success "CUDA环境检查通过"
    else
        log_error "未检测到CUDA环境"
        exit 1
    fi
}

# 系统更新
update_system() {
    log_info "更新系统包..."
    sudo apt update
    sudo apt upgrade -y
    log_success "系统更新完成"
}

# 安装基础依赖
install_basic_deps() {
    log_info "安装基础依赖..."
    sudo apt install -y \
        curl wget git vim htop tree \
        build-essential cmake pkg-config \
        python3-pip python3-dev python3-venv \
        libopencv-dev python3-opencv \
        libfreetype6-dev libpng-dev \
        libjpeg-dev libopenblas-dev \
        liblapack-dev gfortran \
        v4l-utils
    log_success "基础依赖安装完成"
}

# 设置Python环境
setup_python_env() {
    log_info "设置Python虚拟环境..."
    
    # 创建虚拟环境
    python3 -m venv sitl_env
    source sitl_env/bin/activate
    
    # 升级pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Python环境设置完成"
}

# 安装PyTorch
install_pytorch() {
    log_info "安装PyTorch (Jetson优化版本)..."
    
    source sitl_env/bin/activate
    
    # 检查JetPack版本
    JETPACK_VERSION=$(dpkg-query --showformat='${Version}' --show nvidia-jetpack 2>/dev/null || echo "unknown")
    log_info "JetPack版本: $JETPACK_VERSION"
    
    # 安装PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # 验证安装
    python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')" || {
        log_error "PyTorch安装验证失败"
        exit 1
    }
    
    log_success "PyTorch安装完成"
}

# 安装项目依赖
install_project_deps() {
    log_info "安装项目依赖..."
    
    source sitl_env/bin/activate
    
    # 安装依赖
    pip install -r requirements_jetson.txt
    
    log_success "项目依赖安装完成"
}

# 优化系统设置
optimize_system() {
    log_info "优化系统设置..."
    
    # 设置功耗模式为平衡模式
    sudo nvpmodel -m 1
    log_info "设置功耗模式为平衡模式 (15W)"
    
    # 启用最大时钟频率
    sudo jetson_clocks
    log_info "启用最大时钟频率"
    
    # 优化内存设置
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    echo 'vm.dirty_ratio=5' | sudo tee -a /etc/sysctl.conf
    echo 'vm.dirty_background_ratio=2' | sudo tee -a /etc/sysctl.conf
    
    # 应用设置
    sudo sysctl -p
    
    log_success "系统优化完成"
}

# 创建启动脚本
create_launch_scripts() {
    log_info "创建启动脚本..."
    
    # 创建主启动脚本
    cat > run_sitl.sh << 'EOF'
#!/bin/bash
# SITL任务启动脚本

echo "🚀 启动Jetson SITL任务系统"

# 激活虚拟环境
source sitl_env/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OPENCV_DNN_BACKEND=CUDA
export OPENCV_DNN_TARGET=CUDA

# 检查GPU状态
echo "GPU状态:"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# 设置功耗模式
echo "设置功耗模式..."
sudo nvpmodel -m 1
sudo jetson_clocks

# 运行程序
echo "启动SITL任务..."
python3 dual_thread_sitl_mission_jetson.py "$@"
EOF
    
    chmod +x run_sitl.sh
    
    # 创建监控脚本
    cat > monitor_system.sh << 'EOF'
#!/bin/bash
# 系统监控脚本

echo "🔍 Jetson系统监控"
echo "按Ctrl+C退出"

while true; do
    clear
    echo "=== Jetson系统状态 $(date) ==="
    echo
    
    # CPU温度和使用率
    CPU_TEMP=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{print $1/1000}' || echo "N/A")
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    echo "CPU: ${CPU_TEMP}°C, 使用率: ${CPU_USAGE}"
    
    # GPU温度
    GPU_TEMP=$(cat /sys/class/thermal/thermal_zone1/temp 2>/dev/null | awk '{print $1/1000}' || echo "N/A")
    echo "GPU: ${GPU_TEMP}°C"
    
    # 内存使用
    echo "内存使用:"
    free -h | grep -E "(Mem|Swap)"
    
    # GPU状态
    echo
    echo "GPU状态:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU信息不可用"
    
    # 功耗模式
    echo
    echo "功耗模式:"
    sudo nvpmodel -q 2>/dev/null | grep "NV Power Mode" || echo "功耗信息不可用"
    
    echo "=========================="
    sleep 3
done
EOF
    
    chmod +x monitor_system.sh
    
    # 创建测试脚本
    cat > test_installation.sh << 'EOF'
#!/bin/bash
# 安装测试脚本

echo "🧪 测试Jetson SITL安装"

# 激活虚拟环境
source sitl_env/bin/activate

echo "1. 测试Python环境..."
python3 -c "
import sys
print(f'Python版本: {sys.version}')
print('✅ Python环境正常')
"

echo "2. 测试依赖库..."
python3 -c "
try:
    import torch
    import cv2
    import numpy as np
    import tensorrt as trt
    import ultralytics
    import easyocr
    print('✅ 所有依赖库导入成功')
    print(f'PyTorch版本: {torch.__version__}')
    print(f'OpenCV版本: {cv2.__version__}')
    print(f'TensorRT版本: {trt.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ 依赖库导入失败: {e}')
    exit(1)
"

echo "3. 测试GPU..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "4. 测试摄像头..."
if ls /dev/video* >/dev/null 2>&1; then
    echo "✅ 检测到摄像头设备:"
    ls /dev/video*
else
    echo "⚠️ 未检测到摄像头设备"
fi

echo "✅ 安装测试完成"
EOF
    
    chmod +x test_installation.sh
    
    log_success "启动脚本创建完成"
}

# 创建配置文件
create_config_files() {
    log_info "创建配置文件..."
    
    # 创建Jetson配置文件
    cat > config_jetson.py << 'EOF'
#!/usr/bin/env python3
# Jetson Orin Nano优化配置

JETSON_CONFIG = {
    # 模型配置
    'model_path': 'weights/best1.pt',
    'confidence_threshold': 0.25,
    'use_tensorrt': True,
    
    # 队列配置
    'detection_queue_size': 300,
    'result_queue_size': 150,
    'queue_wait_timeout': 3.0,
    
    # 性能配置
    'max_fps': 30,
    'memory_cleanup_interval': 50,
    'thermal_check_interval': 100,
    
    # 功耗配置
    'power_mode': 'balanced',  # power_save, balanced, performance
    
    # 摄像头配置
    'camera_width': 1920,
    'camera_height': 1080,
    'camera_fps': 30,
    'camera_device': 0,  # 0为默认摄像头
    
    # 显示配置
    'display_width': 1280,
    'display_height': 720,
    'show_processing_window': True,
    
    # 存储配置
    'save_raw_detections': True,
    'save_processing_results': True,
    'data_save_interval': 100,  # 每100帧保存一次
    
    # 调试配置
    'debug_mode': False,
    'verbose_logging': True,
}

# 摄像头配置选项
CAMERA_CONFIGS = {
    'usb_camera': {
        'source': 0,
        'backend': 'v4l2'
    },
    'csi_camera': {
        'source': 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink',
        'backend': 'gstreamer'
    }
}
EOF
    
    log_success "配置文件创建完成"
}

# 设置权限和环境
setup_permissions() {
    log_info "设置权限和环境..."
    
    # 添加用户到video组
    sudo usermod -a -G video $USER
    
    # 设置GPIO权限（如果需要）
    sudo usermod -a -G gpio $USER 2>/dev/null || true
    
    # 创建数据目录
    mkdir -p data logs weights
    
    log_success "权限和环境设置完成"
}

# 下载示例模型（可选）
download_example_model() {
    log_info "是否下载示例YOLO模型? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "下载示例模型..."
        mkdir -p weights
        
        # 这里可以下载预训练模型
        # wget -O weights/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
        
        log_info "请手动将您的模型文件放置到 weights/ 目录"
    fi
}

# 主安装流程
main() {
    echo "开始安装过程..."
    
    check_jetson
    check_requirements
    update_system
    install_basic_deps
    setup_python_env
    install_pytorch
    install_project_deps
    optimize_system
    create_launch_scripts
    create_config_files
    setup_permissions
    download_example_model
    
    log_success "🎉 Jetson SITL项目安装完成!"
    echo
    echo "下一步操作:"
    echo "1. 将您的YOLO模型文件放置到 weights/ 目录"
    echo "2. 运行测试: ./test_installation.sh"
    echo "3. 启动系统: ./run_sitl.sh"
    echo "4. 监控系统: ./monitor_system.sh"
    echo
    echo "重要提示:"
    echo "- 首次运行可能需要转换TensorRT引擎，请耐心等待"
    echo "- 确保摄像头正确连接"
    echo "- 监控温度，避免过热"
    echo
    log_success "安装完成，享受您的Jetson SITL系统!"
}

# 错误处理
trap 'log_error "安装过程中发生错误，请检查日志"; exit 1' ERR

# 运行主函数
main "$@" 