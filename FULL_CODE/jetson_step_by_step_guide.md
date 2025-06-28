# Jetson Orin Nano 配置操作指南 - 一步步教程

## 🎯 开始之前的准备工作

### 检查您的硬件和系统
```bash
# 1. 检查Jetson型号和JetPack版本
cat /etc/nv_tegra_release

# 2. 检查系统信息
uname -a
lsb_release -a

# 3. 检查可用空间（至少需要10GB）
df -h

# 4. 检查内存
free -h
```

**预期输出示例：**
```
# Tegra版本应该显示类似：R35 (release), REVISION: 4.1
# Ubuntu版本应该是20.04 LTS
# 可用空间应该大于10GB
```

---

## 📋 第一步：系统基础配置

### 1.1 更新系统包
```bash
# 更新包列表
sudo apt update

# 升级所有包（这可能需要5-10分钟）
sudo apt upgrade -y

# 安装基础工具
sudo apt install -y curl wget git vim htop tree build-essential
```

**验证：**
```bash
# 检查工具是否安装成功
git --version
curl --version
```

### 1.2 设置性能模式
```bash
# 查看当前功耗模式
sudo nvpmodel -q

# 设置为最大性能模式（模式0 - MAXN）
sudo nvpmodel -m 0

# 锁定最高时钟频率
sudo jetson_clocks

# 验证设置（让它运行几秒钟然后按Ctrl+C停止）
sudo tegrastats
```

**预期输出：**
```
NV Power Mode: MAXN
```

---

## 🐍 第二步：安装Conda（如果还没有）

### 2.1 检查是否已安装Conda
```bash
# 检查conda是否存在
conda --version
```

### 2.2 如果没有安装，下载并安装Miniconda
```bash
# 下载Miniconda for ARM64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

# 安装Miniconda
bash Miniconda3-latest-Linux-aarch64.sh

# 重新加载bash配置
source ~/.bashrc

# 验证安装
conda --version
```

**验证：**
```bash
# 应该显示conda版本号，如：conda 23.x.x
```

---

## 🔧 第三步：创建和配置Conda环境

### 3.1 创建专用环境
```bash
# 创建名为jetson_yolo的环境
conda create -n jetson_yolo python=3.8 -y

# 激活环境
conda activate jetson_yolo

# 验证环境
python --version
which python
```

**验证：**
```bash
# Python版本应该显示：Python 3.8.x
# 路径应该包含：/miniconda3/envs/jetson_yolo/
```

### 3.2 安装基础科学计算包
```bash
# 使用conda安装基础包
conda install -y numpy=1.21.0 scipy matplotlib pillow scikit-image seaborn

# 验证安装
python -c "import numpy; print('NumPy版本:', numpy.__version__)"
python -c "import scipy; print('SciPy版本:', scipy.__version__)"
python -c "import matplotlib; print('Matplotlib版本:', matplotlib.__version__)"
```

**预期输出：**
```
NumPy版本: 1.21.0
SciPy版本: 1.x.x
Matplotlib版本: 3.x.x
```

---

## 🔥 第四步：安装CUDA和深度学习框架

### 4.1 验证CUDA环境
```bash
# 检查CUDA版本
nvcc --version

# 检查CUDA路径
ls /usr/local/cuda/

# 设置CUDA环境变量
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc
```

### 4.2 安装系统依赖
```bash
# 安装PyTorch编译依赖
sudo apt install -y libopenblas-base libopenmpi-dev libomp-dev

# 安装图像处理依赖
sudo apt install -y libjpeg-dev zlib1g-dev

# 安装OCR相关依赖
sudo apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

### 4.3 下载并安装PyTorch for Jetson
```bash
# 确保在jetson_yolo环境中
conda activate jetson_yolo

# 下载PyTorch wheel文件（这可能需要几分钟）
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl -O torch-2.0.0-cp38-cp38-linux_aarch64.whl

# 安装PyTorch
pip install torch-2.0.0-cp38-cp38-linux_aarch64.whl

# 验证PyTorch安装
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
python -c "import torch; print('CUDA设备数:', torch.cuda.device_count())"
```

**预期输出：**
```
PyTorch版本: 2.0.0+nv23.05
CUDA可用: True
CUDA设备数: 1
```

### 4.4 安装torchvision
```bash
# 克隆torchvision源码
git clone --branch v0.15.1 https://github.com/pytorch/vision torchvision

# 进入目录并编译安装（这可能需要10-15分钟）
cd torchvision
export BUILD_VERSION=0.15.1
python setup.py install --user

# 返回上级目录
cd ..

# 验证torchvision
python -c "import torchvision; print('torchvision版本:', torchvision.__version__)"
```

### 4.5 配置TensorRT（重要！）

#### 4.5.1 验证TensorRT安装
```bash
# 检查TensorRT是否已安装（JetPack通常预装）
dpkg -l | grep tensorrt

# 检查TensorRT版本
/usr/src/tensorrt/bin/trtexec --help | head -5

# 验证Python TensorRT绑定
python -c "import tensorrt; print('TensorRT版本:', tensorrt.__version__)"
```

#### 4.5.2 安装TensorRT Python依赖
```bash
# 确保在jetson_yolo环境中
conda activate jetson_yolo

# 安装pycuda（TensorRT推理必需）
pip install pycuda

# 安装torch2trt（PyTorch到TensorRT转换工具）
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install --user
cd ..

# 验证torch2trt安装
python -c "import torch2trt; print('✅ torch2trt安装成功')"
```

#### 4.5.3 测试TensorRT功能
```bash
# 创建TensorRT测试脚本
cat << 'EOF' > test_tensorrt.py
import torch
import tensorrt as trt
import numpy as np
import time

print("⚡ 测试TensorRT功能...")

# 检查TensorRT版本
print(f"TensorRT版本: {trt.__version__}")

# 测试基本TensorRT功能
try:
    # 创建TensorRT Logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 创建Builder
    builder = trt.Builder(logger)
    print("✅ TensorRT Builder创建成功")
    
    # 创建Network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    print("✅ TensorRT Network创建成功")
    
    print("✅ TensorRT基本功能测试通过")
    
except Exception as e:
    print(f"❌ TensorRT测试失败: {e}")

# 测试torch2trt转换
print("\n🔄 测试torch2trt转换...")
try:
    from torch2trt import torch2trt
    
    # 创建简单模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
            
        def forward(self, x):
            return torch.relu(self.conv(x))
    
    # 创建模型和示例输入
    model = SimpleModel().eval().cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    
    # 转换为TensorRT
    print("🔄 正在转换模型到TensorRT...")
    model_trt = torch2trt(model, [x], fp16_mode=True)
    print("✅ 模型转换成功")
    
    # 比较推理时间
    # 原始PyTorch推理
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            y_torch = model(x)
    torch.cuda.synchronize()
    time_torch = (time.time() - start) / 100
    
    # TensorRT推理
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y_trt = model_trt(x)
    torch.cuda.synchronize()
    time_trt = (time.time() - start) / 100
    
    print(f"⏱️  PyTorch推理时间: {time_torch*1000:.2f}ms")
    print(f"⚡ TensorRT推理时间: {time_trt*1000:.2f}ms")
    print(f"🚀 加速比: {time_torch/time_trt:.2f}x")
    
    # 验证输出一致性
    max_diff = torch.max(torch.abs(y_torch - y_trt)).item()
    print(f"📊 最大输出差异: {max_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✅ 输出一致性验证通过")
    else:
        print("⚠️  输出存在较大差异，请检查")
        
except Exception as e:
    print(f"❌ torch2trt测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 TensorRT测试完成")
EOF

# 运行TensorRT测试
python test_tensorrt.py
```

#### 4.5.4 安装YOLO TensorRT支持
```bash
# 安装YOLOv5的TensorRT导出依赖
pip install onnx onnxruntime-gpu

# 克隆YOLOv5仓库（如果还没有）
if [ ! -d "yolov5" ]; then
    git clone https://github.com/ultralytics/yolov5.git
fi

# 测试ONNX导出功能
cd yolov5
python export.py --weights yolov5s.pt --include onnx --device 0
cd ..
```

**预期输出：**
```
TensorRT版本: 8.5.x
✅ TensorRT Builder创建成功
✅ TensorRT Network创建成功
🚀 加速比: 2.0x以上
```

---

## 📷 第五步：配置OpenCV和摄像头

### 5.1 验证OpenCV
```bash
# 检查系统OpenCV版本
python -c "import cv2; print('OpenCV版本:', cv2.__version__)"
python -c "import cv2; print('CUDA设备数量:', cv2.cuda.getCudaEnabledDeviceCount())"
```

### 5.2 测试摄像头
```bash
# 创建摄像头测试脚本
cat << 'EOF' > test_camera.py
import cv2
import sys

print("🔍 测试摄像头连接...")

# 测试USB摄像头
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ USB摄像头 (索引0) 可用")
    ret, frame = cap.read()
    if ret:
        print(f"✅ 图像尺寸: {frame.shape}")
        print("✅ 摄像头测试成功")
    else:
        print("❌ 无法读取图像")
    cap.release()
else:
    print("❌ USB摄像头不可用，尝试其他索引...")
    # 尝试其他摄像头索引
    for i in range(1, 4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ 找到摄像头在索引 {i}")
            cap.release()
            break
    else:
        print("❌ 未找到可用的USB摄像头")

print("📷 摄像头测试完成")
EOF

# 运行测试
python test_camera.py
```

---

## 🔤 第六步：安装OCR环境

### 6.1 安装EasyOCR
```bash
# 安装EasyOCR
pip install easyocr

# 验证安装
python -c "import easyocr; print('✅ EasyOCR导入成功')"
```

### 6.2 预下载OCR模型
```bash
# 创建OCR测试脚本
cat << 'EOF' > test_ocr.py
import easyocr
import numpy as np
import time

print("🔤 初始化EasyOCR...")
start_time = time.time()

try:
    # 尝试GPU模式
    reader = easyocr.Reader(['en'], gpu=True, download_enabled=True)
    print("✅ EasyOCR GPU模式初始化成功")
except Exception as e:
    print(f"⚠️  GPU模式失败: {e}")
    try:
        # 回退到CPU模式
        reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        print("✅ EasyOCR CPU模式初始化成功")
    except Exception as e:
        print(f"❌ OCR初始化失败: {e}")
        exit(1)

init_time = time.time() - start_time
print(f"⏱️  初始化用时: {init_time:.2f}秒")

# 测试识别
print("🧪 测试OCR识别...")
test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
results = reader.readtext(test_image)
print("✅ OCR测试完成")
print("🎉 EasyOCR配置成功")
EOF

# 运行测试（首次运行会下载模型，需要几分钟）
python test_ocr.py
```

---

## 🎯 第七步：安装YOLO环境

### 7.1 安装YOLO相关包
```bash
# 安装ultralytics和相关依赖
pip install ultralytics
pip install seaborn thop

# 安装其他必需包
pip install pyclipper shapely pymavlink pycuda
```

### 7.2 测试YOLO环境
```bash
# 创建YOLO测试脚本
cat << 'EOF' > test_yolo.py
import torch
import numpy as np
import time

print("🎯 测试YOLO环境...")

# 检查基本信息
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# 测试GPU推理
print("🧪 测试GPU推理性能...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建测试张量
x = torch.randn(1, 3, 640, 640).to(device)

# 测试推理速度
start_time = time.time()
for i in range(10):
    y = torch.nn.functional.relu(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
inference_time = (time.time() - start_time) / 10
print(f"⏱️  平均推理时间: {inference_time*1000:.2f}ms")

print("✅ YOLO环境测试成功")
EOF

# 运行测试
python test_yolo.py
```

---

## 🔧 第八步：系统优化配置

### 8.1 增加Swap空间（重要！）
```bash
# 检查当前swap
free -h

# 创建4GB swap文件
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 设置开机自动挂载
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 验证swap
free -h
swapon --show
```

### 8.2 设置永久性能模式
```bash
# 创建开机自动设置脚本
sudo tee /etc/systemd/system/jetson-performance.service << 'EOF'
[Unit]
Description=Jetson Performance Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/nvpmodel -m 0
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# 启用服务
sudo systemctl enable jetson-performance.service
sudo systemctl start jetson-performance.service

# 验证服务状态
sudo systemctl status jetson-performance.service
```

---

## 🧪 第九步：完整环境测试

### 9.1 创建综合测试脚本
```bash
cat << 'EOF' > final_test.py
#!/usr/bin/env python3
import sys
import time
import traceback

def test_imports():
    """测试所有必要的包导入"""
    print("📦 测试包导入...")
    
    tests = [
        ("Python", lambda: sys.version),
        ("NumPy", lambda: __import__('numpy').__version__),
        ("SciPy", lambda: __import__('scipy').__version__),
        ("PyTorch", lambda: __import__('torch').__version__),
        ("OpenCV", lambda: __import__('cv2').__version__),
        ("EasyOCR", lambda: __import__('easyocr') and "OK"),
        ("Ultralytics", lambda: __import__('ultralytics') and "OK"),
        ("PyCUDA", lambda: __import__('pycuda') and "OK"),
        ("TensorRT", lambda: __import__('tensorrt').__version__),
        ("torch2trt", lambda: __import__('torch2trt') and "OK"),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            result = test_func()
            print(f"✅ {name}: {result}")
            results[name] = True
        except Exception as e:
            print(f"❌ {name}: {e}")
            results[name] = False
    
    return results

def test_cuda():
    """测试CUDA功能"""
    print("\n🔥 测试CUDA功能...")
    
    try:
        import torch
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数: {torch.cuda.device_count()}")
            print(f"当前设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name(0)}")
            
            # 测试GPU内存
            device = torch.device('cuda')
            x = torch.randn(1000, 1000).to(device)
            y = torch.mm(x, x.t())
            print("✅ GPU计算测试成功")
            return True
    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")
        return False

def test_camera():
    """测试摄像头"""
    print("\n📷 测试摄像头...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ 摄像头工作正常，图像尺寸: {frame.shape}")
                cap.release()
                return True
            else:
                print("❌ 无法读取摄像头图像")
                cap.release()
                return False
        else:
            print("❌ 无法打开摄像头")
            return False
    except Exception as e:
        print(f"❌ 摄像头测试失败: {e}")
        return False

def test_ocr():
    """测试OCR功能"""
    print("\n🔤 测试OCR功能...")
    
    try:
        import easyocr
        import numpy as np
        
        # 尝试GPU模式
        try:
            reader = easyocr.Reader(['en'], gpu=True)
            print("✅ OCR GPU模式初始化成功")
        except:
            reader = easyocr.Reader(['en'], gpu=False)
            print("✅ OCR CPU模式初始化成功")
        
        # 测试识别
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        results = reader.readtext(test_image)
        print("✅ OCR功能测试成功")
        return True
        
    except Exception as e:
        print(f"❌ OCR测试失败: {e}")
        return False

def test_tensorrt():
    """测试TensorRT功能"""
    print("\n⚡ 测试TensorRT功能...")
    
    try:
        import tensorrt as trt
        import torch
        from torch2trt import torch2trt
        
        print(f"TensorRT版本: {trt.__version__}")
        
        # 测试基本TensorRT功能
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        print("✅ TensorRT基本功能正常")
        
        # 测试简单模型转换
        class TestModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)
        
        model = TestModel().eval().cuda()
        x = torch.randn(1, 3, 32, 32).cuda()
        
        # 快速转换测试
        model_trt = torch2trt(model, [x], max_batch_size=1)
        y_trt = model_trt(x)
        
        print("✅ TensorRT模型转换和推理成功")
        return True
        
    except Exception as e:
        print(f"❌ TensorRT测试失败: {e}")
        return False

def main():
    print("🧪 开始完整环境测试...\n")
    
    # 运行所有测试
    import_results = test_imports()
    cuda_ok = test_cuda()
    camera_ok = test_camera()
    ocr_ok = test_ocr()
    tensorrt_ok = test_tensorrt()
    
    # 总结结果
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    print("="*50)
    
    total_tests = len(import_results) + 4  # 导入测试 + CUDA + 摄像头 + OCR + TensorRT
    passed_tests = sum(import_results.values()) + sum([cuda_ok, camera_ok, ocr_ok, tensorrt_ok])
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！环境配置成功！")
        print("您现在可以运行YOLO+OCR推理代码了。")
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败，请检查配置。")
    
    print("\n💡 激活环境命令: conda activate jetson_yolo")

if __name__ == "__main__":
    main()
EOF

# 运行最终测试
python final_test.py
```

---

## ✅ 第十步：验证和清理

### 10.1 创建环境激活脚本
```bash
# 创建便捷的环境激活脚本
cat << 'EOF' > activate_jetson_env.sh
#!/bin/bash
echo "🚀 激活Jetson YOLO环境..."

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate jetson_yolo

# 显示环境信息
echo "✅ 环境已激活："
echo "Python: $(python --version)"
echo "环境路径: $(which python)"

# 设置性能模式（如果需要）
echo "🔧 检查性能模式..."
sudo nvpmodel -q

echo "🎯 环境准备完成！可以开始运行代码了。"
EOF

chmod +x activate_jetson_env.sh
```

### 10.2 清理临时文件
```bash
# 清理下载的文件
rm -f torch-2.0.0-cp38-cp38-linux_aarch64.whl
rm -f Miniconda3-latest-Linux-aarch64.sh

# 清理编译目录（可选，如果需要节省空间）
# rm -rf torchvision

echo "🧹 清理完成"
```

### 10.3 保存环境配置
```bash
# 导出conda环境配置
conda env export > jetson_yolo_environment.yml

# 创建pip requirements文件
pip freeze > requirements_jetson.txt

echo "💾 环境配置已保存"
```

---

## 🎉 完成！

**恭喜！您已经完成了Jetson Orin Nano的完整环境配置。**

### 下次使用时的快速启动：
```bash
# 激活环境
conda activate jetson_yolo

# 或使用脚本
./activate_jetson_env.sh

# 检查性能模式
sudo nvpmodel -q

# 运行您的代码
python your_inference_script.py
```

### 如果遇到问题：
1. **重新运行测试**：`python final_test.py`
2. **检查环境**：`conda list`
3. **查看系统资源**：`htop` 和 `sudo tegrastats`
4. **重新激活环境**：`conda deactivate && conda activate jetson_yolo`

**您的Jetson现在已经准备好运行高性能的YOLO+OCR推理任务了！** 🚀 