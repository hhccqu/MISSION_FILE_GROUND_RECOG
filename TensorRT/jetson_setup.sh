#!/bin/bash
# jetson_setup.sh
# Jetson Orin Nano 自动化配置脚本

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# 检查是否在Jetson设备上运行
check_jetson() {
    print_header "检查Jetson设备"
    
    if [ ! -f /proc/device-tree/model ]; then
        print_error "这不是Jetson设备！"
        exit 1
    fi
    
    JETSON_MODEL=$(cat /proc/device-tree/model)
    print_info "检测到设备: $JETSON_MODEL"
    
    if [[ "$JETSON_MODEL" == *"Orin Nano"* ]]; then
        print_success "确认为Jetson Orin Nano"
        JETSON_TYPE="orin_nano"
    elif [[ "$JETSON_MODEL" == *"Orin"* ]]; then
        print_success "确认为Jetson Orin系列"
        JETSON_TYPE="orin"
    elif [[ "$JETSON_MODEL" == *"Xavier"* ]]; then
        print_success "确认为Jetson Xavier系列"
        JETSON_TYPE="xavier"
    else
        print_warning "未识别的Jetson型号，将使用通用配置"
        JETSON_TYPE="generic"
    fi
}

# 检查和设置性能模式
setup_performance_mode() {
    print_header "设置性能模式"
    
    # 检查当前模式
    current_mode=$(sudo nvpmodel -q | grep "NV Power Mode" | awk '{print $NF}')
    print_info "当前功耗模式: $current_mode"
    
    # 设置最高性能模式
    if [[ "$JETSON_TYPE" == "orin_nano" ]]; then
        print_info "设置Orin Nano最高性能模式 (15W)..."
        sudo nvpmodel -m 0
    elif [[ "$JETSON_TYPE" == "orin" ]]; then
        print_info "设置Orin最高性能模式..."
        sudo nvpmodel -m 0
    else
        print_info "设置最高性能模式..."
        sudo nvpmodel -m 0
    fi
    
    # 锁定最高频率
    print_info "锁定最高频率..."
    sudo jetson_clocks
    
    print_success "性能模式设置完成"
}

# 检查JetPack版本
check_jetpack() {
    print_header "检查JetPack版本"
    
    if dpkg -l | grep -q nvidia-jetpack; then
        JETPACK_VERSION=$(dpkg -l | grep nvidia-jetpack | awk '{print $3}')
        print_info "已安装JetPack版本: $JETPACK_VERSION"
    else
        print_warning "未检测到JetPack安装"
        return 1
    fi
    
    # 检查CUDA版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_info "CUDA版本: $CUDA_VERSION"
    else
        print_warning "CUDA未安装或未在PATH中"
        return 1
    fi
    
    # 检查TensorRT版本
    if dpkg -l | grep -q tensorrt; then
        TRT_VERSION=$(dpkg -l | grep "tensorrt " | awk '{print $3}')
        print_info "TensorRT版本: $TRT_VERSION"
    else
        print_warning "TensorRT未安装"
        return 1
    fi
    
    return 0
}

# 安装或更新JetPack
install_jetpack() {
    print_header "安装/更新JetPack"
    
    print_info "更新包列表..."
    sudo apt update
    
    print_info "安装JetPack..."
    sudo apt install -y nvidia-jetpack
    
    print_success "JetPack安装完成"
}

# 安装Python依赖
install_python_deps() {
    print_header "安装Python依赖"
    
    # 更新pip
    print_info "更新pip..."
    python3 -m pip install --upgrade pip
    
    # 安装基础包
    print_info "安装基础Python包..."
    pip3 install --user numpy opencv-python pillow
    
    # 安装深度学习框架
    print_info "安装PyTorch (Jetson版本)..."
    # 使用NVIDIA提供的PyTorch wheel
    if [[ "$JETPACK_VERSION" == *"5.1"* ]]; then
        pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip3 install --user torch torchvision torchaudio
    fi
    
    # 安装Ultralytics YOLO
    print_info "安装Ultralytics YOLO..."
    pip3 install --user ultralytics
    
    # 安装TensorRT Python接口
    print_info "安装PyCUDA..."
    pip3 install --user pycuda
    
    # 安装OCR相关
    print_info "安装EasyOCR..."
    pip3 install --user easyocr
    
    # 安装监控工具
    print_info "安装Jetson监控工具..."
    sudo pip3 install jetson-stats
    
    # 安装其他有用的包
    print_info "安装其他依赖包..."
    pip3 install --user matplotlib seaborn tqdm
    
    print_success "Python依赖安装完成"
}

# 设置环境变量
setup_environment() {
    print_header "设置环境变量"
    
    # 创建环境变量文件
    ENV_FILE="$HOME/.jetson_env"
    
    cat > "$ENV_FILE" << EOF
# Jetson环境变量配置
export CUDA_HOME=/usr/local/cuda
export PATH=\$PATH:\$CUDA_HOME/bin
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CUDA_HOME/lib64
export PYTHONPATH=\$PYTHONPATH:\$HOME/.local/lib/python3.8/site-packages

# TensorRT优化设置
export TRT_LOGGER_LEVEL=1
export CUDA_VISIBLE_DEVICES=0

# OpenCV优化设置
export OPENCV_DNN_CUDA=1
EOF
    
    # 添加到bashrc
    if ! grep -q "source $ENV_FILE" "$HOME/.bashrc"; then
        echo "source $ENV_FILE" >> "$HOME/.bashrc"
        print_info "环境变量已添加到 ~/.bashrc"
    fi
    
    # 立即加载环境变量
    source "$ENV_FILE"
    
    print_success "环境变量设置完成"
}

# 优化系统设置
optimize_system() {
    print_header "优化系统设置"
    
    # 增加swap空间
    if [ ! -f /swapfile ]; then
        print_info "创建4GB swap空间..."
        sudo fallocate -l 4G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        
        # 添加到fstab
        if ! grep -q "/swapfile" /etc/fstab; then
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        fi
        print_success "Swap空间创建完成"
    else
        print_info "Swap空间已存在"
    fi
    
    # 优化GPU内存设置
    print_info "优化GPU设置..."
    if [ ! -f /etc/systemd/system/jetson-gpu-optimize.service ]; then
        sudo tee /etc/systemd/system/jetson-gpu-optimize.service > /dev/null << EOF
[Unit]
Description=Jetson GPU Optimization
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo 1 > /sys/devices/gpu.0/power_control'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl enable jetson-gpu-optimize.service
        print_success "GPU优化服务已创建"
    fi
    
    # 设置CPU调度器
    print_info "优化CPU调度器..."
    echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    
    print_success "系统优化完成"
}

# 转换YOLO模型为TensorRT
convert_yolo_model() {
    print_header "转换YOLO模型为TensorRT"
    
    # 查找模型文件
    MODEL_DIRS=("../weights" "weights" "../ready/weights")
    MODEL_PATH=""
    
    for dir in "${MODEL_DIRS[@]}"; do
        if [ -f "$dir/best1.pt" ]; then
            MODEL_PATH="$dir/best1.pt"
            break
        fi
    done
    
    if [ -z "$MODEL_PATH" ]; then
        print_warning "未找到YOLO模型文件 (best1.pt)"
        print_info "请将模型文件放置在以下目录之一:"
        for dir in "${MODEL_DIRS[@]}"; do
            print_info "  - $dir/"
        done
        return 1
    fi
    
    print_info "找到模型文件: $MODEL_PATH"
    
    # 转换为TensorRT引擎
    ENGINE_PATH="${MODEL_PATH%.*}.engine"
    
    if [ ! -f "$ENGINE_PATH" ]; then
        print_info "转换为TensorRT引擎..."
        
        # 使用Python脚本转换
        python3 << EOF
import sys
try:
    from ultralytics import YOLO
    
    print("加载YOLO模型...")
    model = YOLO("$MODEL_PATH")
    
    print("转换为TensorRT引擎 (FP16模式)...")
    model.export(
        format='engine',
        half=True,
        device=0,
        workspace=2,
        verbose=True
    )
    
    print("转换完成!")
    
except Exception as e:
    print(f"转换失败: {e}")
    sys.exit(1)
EOF
        
        if [ $? -eq 0 ]; then
            print_success "TensorRT引擎转换完成: $ENGINE_PATH"
        else
            print_error "TensorRT引擎转换失败"
            return 1
        fi
    else
        print_info "TensorRT引擎已存在: $ENGINE_PATH"
    fi
    
    return 0
}

# 运行性能测试
run_performance_test() {
    print_header "运行性能测试"
    
    if [ -f "performance_test.py" ]; then
        print_info "运行性能对比测试..."
        python3 performance_test.py
    else
        print_warning "未找到性能测试脚本"
    fi
}

# 创建启动脚本
create_launch_script() {
    print_header "创建启动脚本"
    
    LAUNCH_SCRIPT="run_tensorrt_inference.sh"
    
    cat > "$LAUNCH_SCRIPT" << 'EOF'
#!/bin/bash
# TensorRT推理启动脚本

# 设置性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 加载环境变量
source ~/.jetson_env

# 启动TensorRT优化版本
cd "$(dirname "$0")"
python3 inference4_realtime_tensorrt.py

EOF
    
    chmod +x "$LAUNCH_SCRIPT"
    print_success "启动脚本已创建: $LAUNCH_SCRIPT"
}

# 显示系统信息
show_system_info() {
    print_header "系统信息摘要"
    
    echo "设备型号: $(cat /proc/device-tree/model)"
    echo "JetPack版本: $(dpkg -l | grep nvidia-jetpack | awk '{print $3}' || echo '未安装')"
    echo "CUDA版本: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- || echo '未安装')"
    echo "TensorRT版本: $(dpkg -l | grep "tensorrt " | awk '{print $3}' || echo '未安装')"
    echo "当前功耗模式: $(sudo nvpmodel -q | grep "NV Power Mode" | awk '{print $NF}')"
    echo "Python版本: $(python3 --version)"
    echo "PyTorch版本: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
    echo "CUDA可用性: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo '未知')"
}

# 主函数
main() {
    print_header "Jetson Orin Nano TensorRT 配置脚本"
    
    # 检查权限
    if [ "$EUID" -eq 0 ]; then
        print_error "请不要使用root权限运行此脚本"
        exit 1
    fi
    
    # 步骤1: 检查设备
    check_jetson
    
    # 步骤2: 设置性能模式
    setup_performance_mode
    
    # 步骤3: 检查JetPack
    if ! check_jetpack; then
        print_info "需要安装或更新JetPack"
        read -p "是否现在安装JetPack? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_jetpack
        else
            print_warning "跳过JetPack安装"
        fi
    fi
    
    # 步骤4: 安装Python依赖
    print_info "安装Python依赖..."
    read -p "是否安装Python依赖? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_python_deps
    else
        print_warning "跳过Python依赖安装"
    fi
    
    # 步骤5: 设置环境变量
    setup_environment
    
    # 步骤6: 系统优化
    optimize_system
    
    # 步骤7: 转换模型
    convert_yolo_model
    
    # 步骤8: 创建启动脚本
    create_launch_script
    
    # 步骤9: 显示系统信息
    show_system_info
    
    print_header "配置完成"
    print_success "Jetson Orin Nano TensorRT配置已完成!"
    print_info "重启系统以确保所有设置生效:"
    print_info "  sudo reboot"
    print_info ""
    print_info "重启后可以运行:"
    print_info "  ./run_tensorrt_inference.sh"
    print_info ""
    print_info "或者运行性能测试:"
    print_info "  python3 performance_test.py"
}

# 运行主函数
main "$@" 