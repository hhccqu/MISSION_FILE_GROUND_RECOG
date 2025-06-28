#!/usr/bin/env python3
# install_dependencies.py
# Jetson依赖安装脚本

import subprocess
import sys
import os

def run_command(command, description=""):
    """执行命令并显示结果"""
    print(f"🔧 {description}")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误信息: {e.stderr}")
        return False

def install_python_packages():
    """安装Python包"""
    print("\n📦 开始安装Python依赖包...")
    
    packages = [
        ("pip", "更新pip"),
        ("numpy", "NumPy数值计算库"),
        ("opencv-python", "OpenCV计算机视觉库"),
        ("pillow", "PIL图像处理库"),
        ("matplotlib", "Matplotlib绘图库"),
        ("seaborn", "Seaborn统计绘图库"),
        ("tqdm", "进度条库"),
        ("psutil", "系统监控库"),
        ("ultralytics", "Ultralytics YOLO"),
        ("pycuda", "PyCUDA"),
        ("easyocr", "EasyOCR文字识别")
    ]
    
    # 更新pip
    run_command("python3 -m pip install --upgrade pip", "更新pip")
    
    # 安装基础包
    for package, description in packages:
        if package == "pip":
            continue
        
        command = f"pip3 install --user {package}"
        run_command(command, f"安装{description}")

def install_jetson_stats():
    """安装Jetson监控工具"""
    print("\n📊 安装Jetson监控工具...")
    run_command("sudo pip3 install jetson-stats", "安装jetson-stats")

def install_pytorch():
    """安装PyTorch (Jetson版本)"""
    print("\n🔥 安装PyTorch...")
    
    # 检查JetPack版本来确定PyTorch版本
    try:
        result = subprocess.run("dpkg -l | grep nvidia-jetpack", 
                              shell=True, capture_output=True, text=True)
        if "5.1" in result.stdout:
            # JetPack 5.1使用CUDA 11.4
            command = "pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            # 默认安装
            command = "pip3 install --user torch torchvision torchaudio"
        
        run_command(command, "安装PyTorch")
        
    except Exception as e:
        print(f"⚠️  无法检测JetPack版本，使用默认PyTorch安装")
        run_command("pip3 install --user torch torchvision torchaudio", "安装PyTorch")

def setup_environment():
    """设置环境变量"""
    print("\n🌐 设置环境变量...")
    
    bashrc_path = os.path.expanduser("~/.bashrc")
    env_vars = [
        "export CUDA_HOME=/usr/local/cuda",
        "export PATH=$PATH:$CUDA_HOME/bin", 
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64",
        "export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.8/site-packages",
        "export TRT_LOGGER_LEVEL=1",
        "export CUDA_VISIBLE_DEVICES=0",
        "export OPENCV_DNN_CUDA=1"
    ]
    
    # 读取现有bashrc
    try:
        with open(bashrc_path, 'r') as f:
            bashrc_content = f.read()
    except FileNotFoundError:
        bashrc_content = ""
    
    # 添加缺失的环境变量
    added_vars = []
    for var in env_vars:
        if var not in bashrc_content:
            bashrc_content += f"\n{var}"
            added_vars.append(var)
    
    if added_vars:
        with open(bashrc_path, 'w') as f:
            f.write(bashrc_content)
        
        print(f"✅ 已添加 {len(added_vars)} 个环境变量到 ~/.bashrc")
        for var in added_vars:
            print(f"   {var}")
        
        print("⚠️  请运行 'source ~/.bashrc' 或重新登录以加载环境变量")
    else:
        print("✅ 环境变量已经配置完成")

def create_swap_space():
    """创建swap空间"""
    print("\n💾 创建swap空间...")
    
    # 检查是否已有swap
    try:
        result = subprocess.run("swapon --show", shell=True, 
                              capture_output=True, text=True)
        if "/swapfile" in result.stdout:
            print("✅ Swap空间已存在")
            return
    except:
        pass
    
    # 创建4GB swap空间
    commands = [
        ("sudo fallocate -l 4G /swapfile", "创建swap文件"),
        ("sudo chmod 600 /swapfile", "设置swap文件权限"),
        ("sudo mkswap /swapfile", "格式化swap文件"),
        ("sudo swapon /swapfile", "启用swap"),
        ("echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab", "添加到fstab")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print("⚠️  Swap创建失败，可能需要手动创建")
            break

def verify_installation():
    """验证安装结果"""
    print("\n🔍 验证安装结果...")
    
    # 运行快速测试
    try:
        from quick_test import quick_test
        success = quick_test()
        return success
    except ImportError:
        print("⚠️  无法运行验证测试")
        return False

def main():
    """主安装流程"""
    print("🚀 Jetson依赖安装脚本")
    print("="*50)
    
    # 检查是否为Jetson设备
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        if 'Jetson' not in model:
            print("❌ 这不是Jetson设备！")
            sys.exit(1)
        print(f"✅ 检测到Jetson设备: {model}")
    except:
        print("❌ 无法检测设备型号")
        sys.exit(1)
    
    # 询问用户要安装哪些组件
    print("\n请选择要安装的组件:")
    print("1. Python依赖包")
    print("2. PyTorch")
    print("3. Jetson监控工具")
    print("4. 环境变量配置")
    print("5. Swap空间")
    print("6. 全部安装")
    
    choice = input("\n请输入选择 (1-6): ").strip()
    
    if choice == "1":
        install_python_packages()
    elif choice == "2":
        install_pytorch()
    elif choice == "3":
        install_jetson_stats()
    elif choice == "4":
        setup_environment()
    elif choice == "5":
        create_swap_space()
    elif choice == "6":
        print("🔧 开始全部安装...")
        install_python_packages()
        install_pytorch()
        install_jetson_stats()
        setup_environment()
        create_swap_space()
        
        print("\n🎉 安装完成！")
        print("请重启终端或运行 'source ~/.bashrc' 加载环境变量")
        
        # 验证安装
        if verify_installation():
            print("✅ 验证成功！环境配置完成。")
        else:
            print("⚠️  部分组件可能需要手动配置")
    else:
        print("❌ 无效选择")
        sys.exit(1)

if __name__ == "__main__":
    main() 