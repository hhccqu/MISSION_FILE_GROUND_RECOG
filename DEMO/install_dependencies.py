#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖安装脚本
自动安装无人机打击任务系统所需的依赖包
"""

import subprocess
import sys
import os

def install_package(package_name, description=""):
    """安装Python包"""
    print(f"📦 安装 {package_name}...")
    if description:
        print(f"   用途: {description}")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package_name} 安装失败: {e}")
        return False

def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """主函数"""
    print("🎯 无人机打击任务系统 - 依赖安装器")
    print("=" * 60)
    
    # 需要安装的包列表
    packages = [
        ("pymavlink", "MAVLink协议通信库，用于与Pixhawk飞控通信"),
        ("numpy", "数值计算库，用于坐标转换和数学运算"),
        ("opencv-python", "计算机视觉库，用于图像处理"),
        ("ultralytics", "YOLO目标检测库"),
        ("easyocr", "OCR文字识别库"),
        ("matplotlib", "绘图库，用于数据可视化"),
        ("statistics", "统计计算库，用于中位数计算")
    ]
    
    # 检查已安装的包
    print("🔍 检查已安装的依赖...")
    installed_packages = []
    missing_packages = []
    
    for package_name, description in packages:
        # 特殊处理一些包名
        import_name = package_name
        if package_name == "opencv-python":
            import_name = "cv2"
        elif package_name == "ultralytics":
            import_name = "ultralytics"
        
        if check_package(import_name):
            print(f"✅ {package_name} - 已安装")
            installed_packages.append(package_name)
        else:
            print(f"❌ {package_name} - 未安装")
            missing_packages.append((package_name, description))
    
    print(f"\n📊 统计:")
    print(f"   已安装: {len(installed_packages)}/{len(packages)}")
    print(f"   需安装: {len(missing_packages)}")
    
    if not missing_packages:
        print("\n🎉 所有依赖都已安装!")
        return
    
    # 询问是否安装缺失的包
    print(f"\n📋 需要安装的包:")
    for package_name, description in missing_packages:
        print(f"   - {package_name}: {description}")
    
    while True:
        choice = input(f"\n❓ 是否安装缺失的依赖? (y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            break
        elif choice in ['n', 'no', '否']:
            print("⚠️ 跳过安装，某些功能可能无法正常使用")
            return
        else:
            print("请输入 y 或 n")
    
    # 安装缺失的包
    print(f"\n🚀 开始安装依赖...")
    success_count = 0
    
    for package_name, description in missing_packages:
        if install_package(package_name, description):
            success_count += 1
        print()  # 空行分隔
    
    # 安装结果
    print(f"📊 安装结果:")
    print(f"   成功: {success_count}/{len(missing_packages)}")
    print(f"   失败: {len(missing_packages) - success_count}")
    
    if success_count == len(missing_packages):
        print(f"\n🎉 所有依赖安装成功!")
        print(f"✅ 系统已准备就绪，可以运行打击任务程序")
    else:
        print(f"\n⚠️ 部分依赖安装失败，请手动安装或检查网络连接")
        print(f"💡 手动安装命令:")
        for package_name, _ in missing_packages:
            print(f"   pip install {package_name}")
    
    # 测试关键功能
    print(f"\n🧪 测试关键功能...")
    
    # 测试pymavlink
    try:
        import pymavlink
        print("✅ MAVLink通信 - 可用")
    except ImportError:
        print("❌ MAVLink通信 - 不可用")
        print("   影响: 无法与飞控通信")
    
    # 测试统计库
    try:
        import statistics
        print("✅ 统计计算 - 可用")
    except ImportError:
        print("❌ 统计计算 - 不可用")
        print("   影响: 无法计算中位数")
    
    # 测试JSON
    try:
        import json
        print("✅ JSON处理 - 可用")
    except ImportError:
        print("❌ JSON处理 - 不可用")
        print("   影响: 无法读取目标数据")
    
    print(f"\n✅ 依赖检查完成!")

if __name__ == "__main__":
    main() 