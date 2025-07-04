#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装PaddleOCR测试所需的依赖包
"""

import subprocess
import sys
import os

def install_package(package):
    """安装单个包"""
    try:
        print(f"📦 正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 安装失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始安装PaddleOCR测试依赖")
    print("=" * 50)
    
    # 需要安装的包列表
    packages = [
        "paddlepaddle",
        "paddleocr",
        "opencv-python",
        "matplotlib",
        "numpy",
        "Pillow"
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"📊 安装完成: {success_count}/{total_count} 个包安装成功")
    
    if success_count == total_count:
        print("🎉 所有依赖安装成功！可以运行PaddleOCR测试了")
        print("\n运行命令:")
        print("python DEMO_DETECT_TEST/paddle_ocr_test.py")
    else:
        print("⚠️  部分依赖安装失败，请手动安装失败的包")
        print("\n手动安装命令:")
        for package in packages:
            print(f"pip install {package}")

if __name__ == "__main__":
    main() 