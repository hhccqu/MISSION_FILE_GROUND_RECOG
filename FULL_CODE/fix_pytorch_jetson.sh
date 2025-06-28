#!/bin/bash
# Jetson PyTorch和torchvision兼容性修复脚本

echo "=========================================="
echo "Jetson PyTorch兼容性修复工具"
echo "=========================================="

# 检查是否在conda环境中
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "⚠️  警告: 未检测到conda环境"
    echo "请先激活您的conda环境: conda activate ground_detect"
    exit 1
fi

echo "✅ 当前conda环境: $CONDA_DEFAULT_ENV"

# 检查当前PyTorch版本
echo "📊 检查当前PyTorch和torchvision版本..."
python -c "
try:
    import torch
    import torchvision
    print(f'PyTorch版本: {torch.__version__}')
    print(f'torchvision版本: {torchvision.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'导入错误: {e}')
"

echo ""
echo "🔧 开始修复..."

# 方法1: 尝试安装NVIDIA官方预编译包
echo "方法1: 安装NVIDIA官方PyTorch包..."

# 检测JetPack版本
JETPACK_VERSION=$(dpkg -l | grep nvidia-jetpack | awk '{print $3}' | cut -d. -f1-2)
echo "检测到JetPack版本: $JETPACK_VERSION"

# 卸载现有的PyTorch
echo "🗑️  卸载现有PyTorch包..."
pip uninstall torch torchvision torchaudio -y

# 根据JetPack版本安装对应的PyTorch
if [[ "$JETPACK_VERSION" == "5."* ]]; then
    echo "📦 安装JetPack 5.x兼容的PyTorch..."
    # 对于JetPack 5.x
    wget -q https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
    pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
    pip install torchvision==0.15.0
    rm -f torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
elif [[ "$JETPACK_VERSION" == "4.6"* ]]; then
    echo "📦 安装JetPack 4.6兼容的PyTorch..."
    # 对于JetPack 4.6
    wget -q https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl
    pip install torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl
    pip install torchvision==0.14.0
    rm -f torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl
else
    echo "⚠️  未识别的JetPack版本，尝试安装CPU版本..."
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi

# 验证安装
echo ""
echo "🔍 验证安装结果..."
python -c "
try:
    import torch
    import torchvision
    print(f'✅ PyTorch版本: {torch.__version__}')
    print(f'✅ torchvision版本: {torchvision.__version__}')
    print(f'✅ CUDA可用: {torch.cuda.is_available()}')
    
    # 测试torchvision.ops
    try:
        import torchvision.ops
        print('✅ torchvision.ops 可用')
    except Exception as e:
        print(f'❌ torchvision.ops 错误: {e}')
        
except ImportError as e:
    print(f'❌ 导入错误: {e}')
"

echo ""
echo "🚀 测试YOLO推理..."
python -c "
try:
    from ultralytics import YOLO
    print('✅ ultralytics 可用')
except ImportError:
    print('❌ ultralytics 不可用，请安装: pip install ultralytics')
except Exception as e:
    print(f'⚠️  ultralytics 警告: {e}')
"

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "📋 下一步操作:"
echo "1. 如果仍有问题，运行: python export_onnx.py"
echo "2. 然后使用: python inference4_realtime_fixed.py"
echo "3. 修复版本会自动处理兼容性问题"
echo "" 