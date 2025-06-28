#!/usr/bin/env python3
# quick_test.py
# Jetson环境快速验证脚本

import sys
import os

def quick_test():
    """快速环境测试"""
    print("🚀 Jetson环境快速测试")
    print("="*40)
    
    tests_passed = 0
    total_tests = 0
    
    # 测试1: 检查是否为Jetson设备
    total_tests += 1
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        if 'Jetson' in model:
            print(f"✅ Jetson设备: {model}")
            tests_passed += 1
        else:
            print(f"❌ 非Jetson设备: {model}")
    except:
        print("❌ 无法检测设备型号")
    
    # 测试2: Python版本
    total_tests += 1
    python_version = sys.version.split()[0]
    if python_version >= '3.6':
        print(f"✅ Python版本: {python_version}")
        tests_passed += 1
    else:
        print(f"❌ Python版本过低: {python_version}")
    
    # 测试3: PyTorch和CUDA
    total_tests += 1
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch + CUDA: {torch.__version__}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            tests_passed += 1
        else:
            print(f"❌ PyTorch无CUDA支持: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
    
    # 测试4: TensorRT
    total_tests += 1
    try:
        import tensorrt as trt
        print(f"✅ TensorRT: {trt.__version__}")
        tests_passed += 1
    except ImportError:
        print("❌ TensorRT未安装")
    
    # 测试5: OpenCV
    total_tests += 1
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        tests_passed += 1
    except ImportError:
        print("❌ OpenCV未安装")
    
    # 测试6: Ultralytics YOLO
    total_tests += 1
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO: 已安装")
        tests_passed += 1
    except ImportError:
        print("❌ Ultralytics YOLO未安装")
    
    # 测试7: EasyOCR
    total_tests += 1
    try:
        import easyocr
        print("✅ EasyOCR: 已安装")
        tests_passed += 1
    except ImportError:
        print("❌ EasyOCR未安装")
    
    # 测试8: 模型文件
    total_tests += 1
    model_paths = ["../weights/best.pt", "weights/best.pt", "best.pt"]
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"✅ YOLO模型: {path} ({size_mb:.1f}MB)")
            model_found = True
            tests_passed += 1
            break
    
    if not model_found:
        print("❌ 未找到YOLO模型文件")
    
    # 测试9: Jetson监控工具
    total_tests += 1
    try:
        from jtop import jtop
        print("✅ Jetson监控工具: 已安装")
        tests_passed += 1
    except ImportError:
        print("❌ Jetson监控工具未安装")
    
    # 测试10: 简单CUDA计算
    total_tests += 1
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("✅ CUDA计算测试: 通过")
            tests_passed += 1
        else:
            print("❌ CUDA不可用")
    except Exception as e:
        print(f"❌ CUDA计算测试失败: {e}")
    
    # 结果汇总
    print("\n" + "="*40)
    print(f"📊 测试结果: {tests_passed}/{total_tests} 通过")
    
    if tests_passed == total_tests:
        print("🎉 所有测试通过！环境配置完美！")
        return True
    elif tests_passed >= total_tests * 0.8:
        print("⚠️  大部分测试通过，环境基本可用")
        return True
    else:
        print("❌ 多项测试失败，需要安装缺失组件")
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1) 