# Jetson Orin Nano TensorRT 配置教程

## 📋 目录
1. [系统环境检查](#1-系统环境检查)
2. [JetPack安装配置](#2-jetpack安装配置)
3. [YOLO模型TensorRT转换](#3-yolo模型tensorrt转换)
4. [代码优化](#4-代码优化)
5. [性能测试](#5-性能测试)
6. [故障排除](#6-故障排除)

---

## 1. 系统环境检查

### 1.1 检查Jetson型号和JetPack版本
```bash
# 检查设备型号
cat /proc/device-tree/model

# 检查JetPack版本
sudo apt show nvidia-jetpack

# 检查CUDA版本
nvcc --version

# 检查TensorRT版本
dpkg -l | grep tensorrt
```

### 1.2 系统性能设置
```bash
# 设置最高性能模式 (15W)
sudo nvpmodel -m 0

# 锁定最高频率
sudo jetson_clocks

# 查看当前模式
sudo nvpmodel -q --verbose
```

---

## 2. JetPack安装配置

### 2.1 更新JetPack (如果需要)
```bash
# 更新包列表
sudo apt update

# 安装完整JetPack
sudo apt install nvidia-jetpack

# 或者安装开发版本
sudo apt install nvidia-jetpack-dev
```

### 2.2 安装监控工具
```bash
# 安装jtop系统监控
sudo pip3 install jetson-stats

# 重启后使用
sudo reboot
# 重启后运行
jtop
```

### 2.3 环境变量配置
```bash
# 添加到 ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc

# 重新加载环境变量
source ~/.bashrc
```

---

## 3. YOLO模型TensorRT转换

### 3.1 安装必要的Python包
```bash
# 安装Ultralytics YOLO
pip3 install ultralytics

# 安装TensorRT Python接口
pip3 install pycuda

# 验证安装
python3 -c "import tensorrt as trt; print(f'TensorRT版本: {trt.__version__}')"
```

### 3.2 转换YOLO模型为TensorRT引擎

#### 方法1: 使用Ultralytics (推荐)
```bash
# 进入weights目录
cd weights/

# FP16转换 (推荐)
yolo export model=best.pt format=engine half=True device=0

# INT8转换 (最高性能，需要校准数据)
yolo export model=best.pt format=engine int8=True data=../datasets/your_dataset.yaml device=0
```

#### 方法2: 使用trtexec工具
```bash
# 先转换为ONNX
yolo export model=best.pt format=onnx

# 使用trtexec转换
/usr/src/tensorrt/bin/trtexec \
    --onnx=best.onnx \
    --saveEngine=best_orin_fp16.engine \
    --fp16 \
    --workspace=2048 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640 \
    --verbose
```

### 3.3 验证转换结果
```bash
# 检查生成的引擎文件
ls -la *.engine

# 使用trtexec测试引擎性能
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=best.engine \
    --batch=1 \
    --iterations=100 \
    --avgRuns=10
```

---

## 4. 代码优化

参考 `inference4_realtime_tensorrt.py` 中的优化实现：

### 4.1 主要优化点
- 使用TensorRT引擎进行推理
- 优化内存管理
- 减少不必要的数据拷贝
- 异步处理OCR

### 4.2 性能监控
- 集成jtop监控
- 实时FPS显示
- GPU使用率监控

---

## 5. 性能测试

### 5.1 运行优化后的代码
```bash
# 确保在正确目录
cd LATEST_CODE/

# 运行TensorRT优化版本
python3 inference4_realtime_tensorrt.py
```

### 5.2 性能对比测试
```bash
# 运行性能测试脚本
python3 ../TensorRT/performance_test.py
```

### 5.3 预期性能提升
- **FP32 → FP16**: 50-80% 速度提升
- **FP32 → INT8**: 100-200% 速度提升
- **内存使用**: 减少30-50%

---

## 6. 故障排除

### 6.1 常见问题

#### 问题1: TensorRT版本不兼容
```bash
# 解决方案: 重新安装匹配版本
sudo apt remove --purge nvidia-tensorrt
sudo apt install nvidia-tensorrt-dev
```

#### 问题2: CUDA内存不足
```bash
# 解决方案: 增加swap空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 问题3: 模型转换失败
```bash
# 检查ONNX模型
python3 -c "
import onnx
model = onnx.load('best.onnx')
onnx.checker.check_model(model)
print('ONNX模型验证通过')
"
```

### 6.2 调试工具
```bash
# 使用jtop监控系统资源
jtop

# 检查CUDA设备
python3 -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA设备数量: {torch.cuda.device_count()}')
print(f'当前设备: {torch.cuda.get_device_name(0)}')
"
```

---

## 7. 优化建议

### 7.1 系统级优化
- 使用高速SD卡或NVMe SSD
- 确保散热良好
- 关闭不必要的系统服务

### 7.2 代码级优化
- 使用批处理推理
- 减少Python-CUDA数据传输
- 异步处理非关键任务

---

## 📞 支持

如果遇到问题，请检查：
1. JetPack版本是否为5.1+
2. CUDA和TensorRT是否正确安装
3. 模型文件是否完整
4. 系统资源是否充足

更多信息请参考NVIDIA Jetson官方文档。 