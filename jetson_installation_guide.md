# 🚀 DEMO_SITL项目移植到Jetson Orin Nano完整指南

## 📋 目录
1. [硬件准备](#硬件准备)
2. [系统安装](#系统安装)
3. [环境配置](#环境配置)
4. [依赖安装](#依赖安装)
5. [项目移植](#项目移植)
6. [性能优化](#性能优化)
7. [测试验证](#测试验证)
8. [故障排除](#故障排除)

## 🔧 硬件准备

### Jetson Orin Nano规格
- **GPU**: 1024 CUDA核心，32个Tensor核心
- **AI性能**: 67 TOPS (Sparse), 33 TOPS (Dense)
- **CPU**: 6核Arm Cortex-A78AE @ 1.7GHz
- **内存**: 8GB LPDDR5 @ 102 GB/s
- **存储**: MicroSD + M.2 NVMe SSD支持
- **功耗**: 7W/15W/25W可调模式

### 必需硬件清单
- [ ] Jetson Orin Nano开发板
- [ ] 64GB+ MicroSD卡 (Class 10, U3)
- [ ] USB-C电源适配器 (5V/3A)
- [ ] USB摄像头或CSI摄像头
- [ ] 散热器和风扇
- [ ] 网络连接 (以太网或WiFi)

### 推荐配置
- [ ] M.2 NVMe SSD (256GB+) 用于存储
- [ ] 主动散热解决方案
- [ ] 高质量电源供应

## 💿 系统安装

### Step 1: 下载JetPack SDK
```bash
# 访问NVIDIA官网下载最新JetPack 6.1
# https://developer.nvidia.com/jetpack

# 使用SDK Manager安装（推荐）
# 或下载预构建的SD卡镜像
```

### Step 2: 烧录系统镜像
```bash
# 使用Balena Etcher或dd命令烧录
# Windows: 使用Balena Etcher
# Linux: 
sudo dd if=jetpack_image.img of=/dev/sdX bs=1M status=progress
```

### Step 3: 首次启动配置
```bash
# 连接显示器、键盘、鼠标
# 按照向导完成初始设置
# 创建用户账户
# 连接网络
```

## ⚙️ 环境配置

### Step 1: 系统更新
```bash
# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y curl wget git vim htop tree
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y python3-pip python3-dev python3-venv
```

### Step 2: 设置功耗模式
```bash
# 查看当前功耗模式
sudo nvpmodel -q

# 设置为最大性能模式 (25W)
sudo nvpmodel -m 2

# 设置为平衡模式 (15W)
sudo nvpmodel -m 1

# 设置为省电模式 (7W)
sudo nvpmodel -m 0

# 查看CPU频率
sudo jetson_clocks --show
```

### Step 3: 启用最大性能
```bash
# 启用最大时钟频率
sudo jetson_clocks

# 检查GPU状态
nvidia-smi

# 检查CUDA版本
nvcc --version
```

## 📦 依赖安装

### Step 1: Python环境
```bash
# 创建虚拟环境
python3 -m venv sitl_env
source sitl_env/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel
```

### Step 2: 深度学习框架
```bash
# 安装PyTorch (Jetson专用版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch安装
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### Step 3: 计算机视觉库
```bash
# 安装OpenCV (预编译版本通常已包含CUDA支持)
pip install opencv-python opencv-contrib-python

# 验证OpenCV CUDA支持
python3 -c "import cv2; print(f'OpenCV版本: {cv2.__version__}'); print(f'CUDA设备数: {cv2.cuda.getCudaEnabledDeviceCount()}')"
```

### Step 4: YOLO和TensorRT
```bash
# 安装Ultralytics YOLO
pip install ultralytics

# 安装TensorRT Python包 (如果未预装)
pip install tensorrt

# 验证TensorRT
python3 -c "import tensorrt as trt; print(f'TensorRT版本: {trt.__version__}')"
```

### Step 5: 其他依赖
```bash
# 安装项目依赖
pip install numpy pandas matplotlib seaborn
pip install easyocr  # OCR识别
pip install pymavlink  # MAVLink通信
pip install psutil  # 系统监控
pip install queue-manager  # 队列管理
```

### Step 6: Jetson专用库
```bash
# 安装Jetson Inference (可选，用于额外加速)
sudo apt install -y python3-jetson-inference python3-jetson-utils

# 验证安装
python3 -c "import jetson.inference; print('Jetson Inference库可用')"
```

## 🔄 项目移植

### Step 1: 复制项目文件
```bash
# 创建项目目录
mkdir -p ~/sitl_mission
cd ~/sitl_mission

# 复制项目文件
# - dual_thread_sitl_mission_jetson.py
# - yolo_trt_utils_jetson.py
# - target_geo_calculator.py
# - weights/best1.pt (YOLO模型文件)
```

### Step 2: 创建配置文件
```bash
# 创建配置文件
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
    
    # 显示配置
    'display_width': 1280,
    'display_height': 720,
}
EOF
```

### Step 3: 创建启动脚本
```bash
# 创建启动脚本
cat > run_sitl_jetson.sh << 'EOF'
#!/bin/bash
# Jetson SITL任务启动脚本

echo "🚀 启动Jetson SITL任务系统"

# 激活虚拟环境
source sitl_env/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OPENCV_DNN_BACKEND=CUDA
export OPENCV_DNN_TARGET=CUDA

# 检查GPU状态
echo "GPU状态:"
nvidia-smi

# 设置功耗模式
echo "设置功耗模式为平衡模式..."
sudo nvpmodel -m 1

# 启用最大时钟频率
echo "启用最大时钟频率..."
sudo jetson_clocks

# 运行程序
echo "启动SITL任务..."
python3 dual_thread_sitl_mission_jetson.py

echo "任务完成"
EOF

chmod +x run_sitl_jetson.sh
```

## ⚡ 性能优化

### Step 1: TensorRT模型转换
```python
# 转换YOLO模型为TensorRT引擎
from ultralytics import YOLO

# 加载模型
model = YOLO('weights/best1.pt')

# 导出为TensorRT引擎
model.export(
    format='engine',
    device=0,
    half=True,  # 使用FP16精度
    workspace=4,  # GB
    verbose=True
)

print("✅ TensorRT引擎转换完成")
```

### Step 2: 系统优化脚本
```bash
# 创建优化脚本
cat > optimize_jetson.sh << 'EOF'
#!/bin/bash
# Jetson系统优化脚本

echo "🔧 优化Jetson系统性能..."

# 设置最大性能模式
sudo nvpmodel -m 2
sudo jetson_clocks

# 优化内存管理
echo 1 | sudo tee /proc/sys/vm/overcommit_memory
echo 50 | sudo tee /proc/sys/vm/swappiness

# 优化网络
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf

# 优化文件系统
echo 'vm.dirty_ratio = 5' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio = 2' | sudo tee -a /etc/sysctl.conf

# 应用设置
sudo sysctl -p

echo "✅ 系统优化完成"
EOF

chmod +x optimize_jetson.sh
```

### Step 3: 温度监控脚本
```bash
# 创建温度监控脚本
cat > monitor_temp.sh << 'EOF'
#!/bin/bash
# Jetson温度监控脚本

while true; do
    echo "=== $(date) ==="
    echo "CPU温度: $(cat /sys/class/thermal/thermal_zone0/temp | awk '{print $1/1000}')°C"
    echo "GPU温度: $(cat /sys/class/thermal/thermal_zone1/temp | awk '{print $1/1000}')°C"
    echo "功耗模式: $(sudo nvpmodel -q | grep 'NV Power Mode')"
    echo "内存使用: $(free -h | grep Mem)"
    echo "GPU状态:"
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    echo "------------------------"
    sleep 5
done
EOF

chmod +x monitor_temp.sh
```

## 🧪 测试验证

### Step 1: 基础功能测试
```bash
# 测试Python环境
python3 -c "
import torch
import cv2
import numpy as np
import tensorrt as trt
print('✅ 所有依赖库导入成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'OpenCV版本: {cv2.__version__}')
print(f'TensorRT版本: {trt.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"
```

### Step 2: YOLO模型测试
```python
# 创建测试脚本
cat > test_yolo_jetson.py << 'EOF'
#!/usr/bin/env python3
import time
import cv2
from yolo_trt_utils_jetson import YOLOTRTDetectorJetson

def test_yolo():
    print("🧪 测试YOLO检测器...")
    
    # 初始化检测器
    detector = YOLOTRTDetectorJetson(
        model_path='weights/best1.pt',
        use_trt=True
    )
    
    # 创建测试图像
    test_image = cv2.imread('test_image.jpg')  # 需要准备测试图像
    if test_image is None:
        # 创建随机测试图像
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 预热
    detector.warmup()
    
    # 测试检测
    start_time = time.time()
    detections = detector.detect(test_image)
    end_time = time.time()
    
    print(f"检测结果: {len(detections)}个目标")
    print(f"推理时间: {(end_time - start_time)*1000:.1f}ms")
    
    # 性能统计
    stats = detector.get_performance_stats()
    print(f"平均FPS: {stats.get('fps', 0):.1f}")
    print(f"使用TensorRT: {stats.get('using_tensorrt', False)}")
    
    print("✅ YOLO测试完成")

if __name__ == "__main__":
    test_yolo()
EOF
```

### Step 3: 完整系统测试
```bash
# 运行完整测试
python3 dual_thread_sitl_mission_jetson.py

# 监控系统性能
./monitor_temp.sh &

# 运行一段时间后检查结果
```

## 🔧 故障排除

### 常见问题及解决方案

#### 1. CUDA内存不足
```bash
# 解决方案：减少批处理大小，启用内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 或在代码中设置
torch.cuda.empty_cache()
```

#### 2. TensorRT转换失败
```bash
# 检查TensorRT版本兼容性
python3 -c "import tensorrt as trt; print(trt.__version__)"

# 更新到兼容版本
pip install --upgrade tensorrt
```

#### 3. 温度过高导致降频
```bash
# 检查散热器安装
# 降低功耗模式
sudo nvpmodel -m 1  # 15W模式

# 或添加延时
# 在代码中添加 time.sleep(0.1)
```

#### 4. 摄像头无法打开
```bash
# 检查摄像头设备
ls /dev/video*

# 测试摄像头
v4l2-ctl --list-devices

# 使用GStreamer管道 (CSI摄像头)
# 修改video_source为GStreamer管道字符串
```

#### 5. 内存泄漏
```bash
# 监控内存使用
watch -n 1 free -h

# 在代码中定期清理
gc.collect()
torch.cuda.empty_cache()
```

## 📊 性能基准

### 预期性能指标

| 配置 | FPS | 功耗 | 温度 |
|------|-----|------|------|
| 7W模式 | 15-20 | 7W | 60-70°C |
| 15W模式 | 25-30 | 15W | 70-80°C |
| 25W模式 | 30-35 | 25W | 80-85°C |

### 优化后性能
- **YOLO推理**: 20-30ms (FP16)
- **图像转正**: 5-10ms
- **OCR识别**: 50-100ms
- **总处理延迟**: <150ms
- **内存使用**: <6GB
- **存储需求**: 2-5GB

## 🎯 部署建议

### 生产环境配置
1. **使用NVMe SSD**: 提高I/O性能
2. **主动散热**: 确保稳定运行
3. **UPS电源**: 防止意外断电
4. **网络监控**: 远程监控系统状态
5. **自动重启**: 异常情况下自动恢复

### 维护建议
1. **定期清理**: 清理灰尘，检查散热
2. **系统更新**: 定期更新JetPack和依赖
3. **性能监控**: 监控温度、功耗、性能
4. **数据备份**: 定期备份重要数据
5. **日志记录**: 记录系统运行日志

---

## 📞 技术支持

如遇到问题，请检查：
1. 硬件连接是否正确
2. 软件版本是否兼容
3. 系统资源是否充足
4. 温度是否正常
5. 日志文件中的错误信息

**移植完成后，您将拥有一个在Jetson Orin Nano上高效运行的实时目标检测和处理系统！** 🚀 