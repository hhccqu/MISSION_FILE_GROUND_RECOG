#!/usr/bin/env python3
# conda_env_test.py
# 专门测试conda环境的脚本

import sys
import os
import subprocess

def test_conda_environment():
    """测试conda环境配置"""
    print("🐍 测试conda环境: ground_detect")
    print("="*50)
    
    tests_passed = 0
    total_tests = 0
    missing_packages = []
    
    # 检查是否在conda环境中
    total_tests += 1
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env:
        print(f"✅ Conda环境: {conda_env}")
        tests_passed += 1
    else:
        print("❌ 未检测到conda环境")
    
    # 测试核心包
    packages_to_test = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'easyocr': 'EasyOCR',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'psutil': 'PSUtil'
    }
    
    for package, name in packages_to_test.items():
        total_tests += 1
        try:
            module = __import__(package)
            version = getattr(module, '__version__', '已安装')
            print(f"✅ {name}: {version}")
            tests_passed += 1
        except ImportError:
            print(f"❌ {name}: 未安装")
            missing_packages.append(package)
    
    # 测试CUDA支持
    total_tests += 1
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA支持: 可用")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            tests_passed += 1
        else:
            print("❌ CUDA支持: 不可用")
    except:
        print("❌ CUDA支持: 测试失败")
    
    # 测试缺失的关键包
    critical_missing = {
        'ultralytics': 'Ultralytics YOLO',
        'pycuda': 'PyCUDA',
        'tensorrt': 'TensorRT Python'
    }
    
    print(f"\n🔍 检查缺失的关键包:")
    for package, name in critical_missing.items():
        try:
            __import__(package)
            print(f"✅ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
            missing_packages.append(package)
    
    # 测试Jetson监控工具
    try:
        from jtop import jtop
        print(f"✅ Jetson监控工具: 已安装")
    except ImportError:
        print(f"❌ Jetson监控工具: 未安装")
        missing_packages.append('jetson-stats')
    
    # 结果汇总
    print(f"\n📊 测试结果: {tests_passed}/{total_tests} 通过")
    
    if missing_packages:
        print(f"\n❌ 需要安装的包:")
        for pkg in set(missing_packages):
            print(f"   - {pkg}")
        
        print(f"\n💡 安装命令:")
        print(f"conda activate ground_detect")
        for pkg in set(missing_packages):
            if pkg == 'jetson-stats':
                print(f"sudo pip install {pkg}")
            else:
                print(f"pip install {pkg}")
    
    # 给出建议
    print(f"\n💡 总结:")
    if len(missing_packages) == 0:
        print("🎉 环境完美！可以直接运行项目")
        return True
    elif len(missing_packages) <= 3:
        print("⚠️  环境基本可用，只需安装少量包")
        return True
    else:
        print("❌ 需要安装较多依赖包")
        return False

def generate_install_script():
    """生成安装脚本"""
    script_content = """#!/bin/bash
# conda环境安装脚本
echo "🔧 为ground_detect环境安装缺失组件..."

# 激活conda环境
conda activate ground_detect

# 安装Python包
echo "📦 安装Python包..."
pip install ultralytics
pip install pycuda
pip install pillow

# 安装Jetson监控工具 (需要sudo权限)
echo "📊 安装Jetson监控工具..."
sudo pip install jetson-stats

# 验证安装
echo "🔍 验证安装结果..."
python conda_env_test.py

echo "✅ 安装完成！"
"""
    
    with open('install_for_conda.sh', 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod('install_for_conda.sh', 0o755)
    print(f"\n📄 已生成安装脚本: install_for_conda.sh")
    print(f"运行方式: ./install_for_conda.sh")

if __name__ == "__main__":
    success = test_conda_environment()
    
    if not success:
        generate_install_script()
    
    sys.exit(0 if success else 1) 