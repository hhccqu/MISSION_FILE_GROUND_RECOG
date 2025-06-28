#!/usr/bin/env python3
# cuda_diagnosis.py
# CUDA问题诊断脚本

import sys
import os
import subprocess

def run_command(cmd):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except:
        return False, "", "命令执行失败"

def check_cuda_system():
    """检查系统级CUDA"""
    print("🔧 检查系统级CUDA安装")
    print("-" * 40)
    
    # 检查nvidia-smi
    success, output, error = run_command("nvidia-smi")
    if success:
        print("✅ nvidia-smi 可用")
        print(f"GPU信息:\n{output[:200]}...")
    else:
        print("❌ nvidia-smi 不可用")
        print(f"错误: {error}")
    
    # 检查nvcc
    success, output, error = run_command("nvcc --version")
    if success:
        print("✅ nvcc (CUDA编译器) 可用")
        for line in output.split('\n'):
            if 'release' in line:
                print(f"CUDA版本: {line}")
    else:
        print("❌ nvcc 不可用")
    
    # 检查CUDA路径
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-11.4",
        "/usr/local/cuda-11.8",
        "/usr/local/cuda-12.0"
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✅ 找到CUDA安装: {path}")
            break
    else:
        print("❌ 未找到CUDA安装路径")

def check_pytorch_cuda():
    """检查PyTorch的CUDA支持"""
    print("\n🔥 检查PyTorch CUDA支持")
    print("-" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA编译支持
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用性: {'✅ 可用' if cuda_available else '❌ 不可用'}")
        
        if cuda_available:
            # GPU信息
            device_count = torch.cuda.device_count()
            print(f"GPU数量: {device_count}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {gpu_name}")
            
            # CUDA版本
            cuda_version = torch.version.cuda
            print(f"PyTorch CUDA版本: {cuda_version}")
            
            # 内存信息
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU内存: {memory_total:.1f} GB")
            
            # 简单计算测试
            try:
                x = torch.randn(1000, 1000)
                x_gpu = x.cuda()
                y = torch.mm(x_gpu, x_gpu)
                print("✅ CUDA计算测试: 成功")
            except Exception as e:
                print(f"❌ CUDA计算测试: 失败 - {e}")
        else:
            # 分析为什么CUDA不可用
            print("\n🔍 分析CUDA不可用的原因:")
            
            # 检查是否编译时包含CUDA
            if hasattr(torch.version, 'cuda') and torch.version.cuda is None:
                print("❌ PyTorch编译时未包含CUDA支持")
                print("   解决方案: 安装CUDA版本的PyTorch")
            else:
                print("✅ PyTorch编译时包含CUDA支持")
            
            # 检查CUDA驱动
            success, _, _ = run_command("nvidia-smi")
            if not success:
                print("❌ NVIDIA驱动未安装或不可用")
                print("   解决方案: 安装NVIDIA驱动")
            else:
                print("✅ NVIDIA驱动可用")
            
            # 检查CUDA运行时
            success, _, _ = run_command("nvcc --version")
            if not success:
                print("❌ CUDA工具包未安装")
                print("   解决方案: 安装CUDA Toolkit")
            else:
                print("✅ CUDA工具包已安装")
    
    except ImportError:
        print("❌ PyTorch未安装")

def check_environment_variables():
    """检查环境变量"""
    print("\n🌐 检查CUDA环境变量")
    print("-" * 40)
    
    important_vars = [
        "CUDA_HOME",
        "CUDA_ROOT", 
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES"
    ]
    
    for var in important_vars:
        value = os.environ.get(var, "未设置")
        if var in ["PATH", "LD_LIBRARY_PATH"]:
            if "cuda" in value.lower():
                print(f"✅ {var}: 包含CUDA路径")
            else:
                print(f"⚠️  {var}: 可能缺少CUDA路径")
        else:
            print(f"{'✅' if value != '未设置' else '⚠️ '} {var}: {value}")

def check_jetson_specific():
    """检查Jetson特定配置"""
    print("\n🤖 检查Jetson特定配置")
    print("-" * 40)
    
    # 检查是否为Jetson设备
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        if 'Jetson' in model:
            print(f"✅ Jetson设备: {model}")
            
            # 检查JetPack
            success, output, _ = run_command("dpkg -l | grep nvidia-jetpack")
            if success:
                print("✅ JetPack已安装")
            else:
                print("❌ JetPack未安装")
            
            # 检查功耗模式
            success, output, _ = run_command("sudo nvpmodel -q")
            if success:
                print(f"功耗模式信息:\n{output}")
            else:
                print("⚠️  无法检查功耗模式")
        else:
            print(f"❌ 非Jetson设备: {model}")
    else:
        print("❌ 无法检测设备类型")

def provide_solutions():
    """提供解决方案"""
    print("\n💡 CUDA问题解决方案")
    print("=" * 50)
    
    print("1. 🔧 如果是PyTorch CUDA问题:")
    print("   # 卸载当前PyTorch")
    print("   pip uninstall torch torchvision torchaudio")
    print("   # 安装CUDA版本的PyTorch")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n2. 🌐 如果是环境变量问题:")
    print("   export CUDA_HOME=/usr/local/cuda")
    print("   export PATH=$PATH:$CUDA_HOME/bin")
    print("   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64")
    
    print("\n3. 🚀 如果是Jetson性能模式问题:")
    print("   sudo nvpmodel -m 0  # 设置最高性能模式")
    print("   sudo jetson_clocks   # 锁定最高频率")
    
    print("\n4. 🔄 重启相关服务:")
    print("   # 重新加载环境变量")
    print("   source ~/.bashrc")
    print("   # 或重新登录")

def main():
    """主函数"""
    print("🔍 CUDA问题诊断工具")
    print("=" * 50)
    
    check_cuda_system()
    check_pytorch_cuda()
    check_environment_variables()
    check_jetson_specific()
    provide_solutions()
    
    print("\n📋 诊断完成！")
    print("请根据上述信息解决CUDA问题。")

if __name__ == "__main__":
    main() 