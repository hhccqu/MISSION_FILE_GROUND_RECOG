#!/usr/bin/env python3
# tensorrt_test.py
# TensorRT功能验证脚本 - 针对Jetson设备优化

import sys
import os
import time
import numpy as np
import subprocess
import psutil
import gc
from pathlib import Path

def check_jetson_device():
    """检查是否为Jetson设备"""
    print("🔍 检查设备类型")
    print("-" * 40)
    
    try:
        # 检查设备树文件
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00')
                if 'jetson' in model.lower():
                    print(f"✅ 检测到Jetson设备: {model}")
                    return True, model
        
        # 检查CPU信息
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'tegra' in cpuinfo.lower() or 'nvidia' in cpuinfo.lower():
                print("✅ 检测到Jetson设备 (通过CPU信息)")
                return True, "Jetson设备"
        
        print("⚠️  未检测到Jetson设备，但可能是Jetson")
        return False, "未知设备"
        
    except Exception as e:
        print(f"❌ 设备检查失败: {e}")
        return False, "检查失败"

def check_memory_status():
    """检查内存状态"""
    print("\n💾 内存状态检查")
    print("-" * 40)
    
    # 获取内存信息
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    used_gb = memory.used / (1024**3)
    
    print(f"总内存: {total_gb:.1f} GB")
    print(f"可用内存: {available_gb:.1f} GB")
    print(f"已用内存: {used_gb:.1f} GB")
    print(f"内存使用率: {memory.percent:.1f}%")
    
    if swap.total > 0:
        swap_gb = swap.total / (1024**3)
        swap_used_gb = swap.used / (1024**3)
        print(f"Swap总量: {swap_gb:.1f} GB")
        print(f"Swap已用: {swap_used_gb:.1f} GB")
        print(f"Swap使用率: {swap.percent:.1f}%")
    else:
        print("❌ 未检测到Swap空间")
    
    # 内存建议
    if available_gb < 1.0:
        print("⚠️  可用内存不足1GB，建议释放内存或增加Swap")
        return False
    elif available_gb < 2.0:
        print("⚠️  可用内存较少，建议谨慎进行TensorRT转换")
        return True
    else:
        print("✅ 内存状态良好")
        return True

def setup_swap_if_needed():
    """如果需要，设置Swap空间"""
    print("\n🔄 Swap空间管理")
    print("-" * 40)
    
    swap = psutil.swap_memory()
    memory = psutil.virtual_memory()
    
    if swap.total == 0:
        print("❌ 未检测到Swap空间")
        print("建议创建Swap空间以避免内存不足崩溃")
        
        response = input("是否自动创建4GB Swap文件? (y/n): ")
        if response.lower() == 'y':
            return create_swap_file()
        else:
            print("⚠️  跳过Swap创建，转换时可能因内存不足而失败")
            return False
    else:
        swap_gb = swap.total / (1024**3)
        print(f"✅ 已有Swap空间: {swap_gb:.1f} GB")
        return True

def create_swap_file():
    """创建Swap文件"""
    print("🔧 创建Swap文件...")
    
    try:
        swap_file = "/tmp/tensorrt_swap"
        swap_size = "4G"  # 4GB Swap
        
        # 检查磁盘空间
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < 5:
            print(f"❌ 磁盘空间不足 ({free_gb:.1f}GB)，无法创建Swap")
            return False
        
        # 创建Swap文件
        commands = [
            f"sudo fallocate -l {swap_size} {swap_file}",
            f"sudo chmod 600 {swap_file}",
            f"sudo mkswap {swap_file}",
            f"sudo swapon {swap_file}"
        ]
        
        for cmd in commands:
            print(f"执行: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ 命令失败: {result.stderr}")
                return False
        
        print(f"✅ Swap文件创建成功: {swap_file}")
        return True
        
    except Exception as e:
        print(f"❌ Swap创建失败: {e}")
        return False

def cleanup_swap():
    """清理临时Swap文件"""
    swap_file = "/tmp/tensorrt_swap"
    if os.path.exists(swap_file):
        try:
            subprocess.run(f"sudo swapoff {swap_file}".split(), capture_output=True)
            os.remove(swap_file)
            print(f"🧹 临时Swap文件已清理: {swap_file}")
        except:
            pass

def free_memory():
    """释放内存"""
    print("🧹 释放内存...")
    
    # Python垃圾回收
    gc.collect()
    
    # 清理系统缓存
    try:
        subprocess.run("sudo sync".split(), capture_output=True)
        subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'".split(), capture_output=True)
        print("✅ 系统缓存已清理")
    except:
        print("⚠️  无法清理系统缓存（需要sudo权限）")

def test_tensorrt_import():
    """测试TensorRT导入"""
    print("🔧 测试TensorRT导入")
    print("-" * 40)
    
    try:
        import tensorrt as trt
        print(f"✅ TensorRT版本: {trt.__version__}")
        return True
    except ImportError as e:
        print(f"❌ TensorRT导入失败: {e}")
        print("   解决方案: pip install tensorrt")
        return False

def test_pycuda():
    """测试PyCUDA"""
    print("\n🔥 测试PyCUDA")
    print("-" * 40)
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("✅ PyCUDA导入成功")
        
        # 获取GPU信息
        device = cuda.Device(0)
        print(f"GPU名称: {device.name()}")
        print(f"计算能力: {device.compute_capability()}")
        
        # 获取GPU内存信息
        free_mem, total_mem = cuda.mem_get_info()
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        
        print(f"GPU内存: {free_gb:.1f}GB 可用 / {total_gb:.1f}GB 总计")
        
        if free_gb < 0.5:
            print("⚠️  GPU内存不足，可能影响TensorRT转换")
        
        return True
    except Exception as e:
        print(f"❌ PyCUDA测试失败: {e}")
        return False

def test_yolo_model_exists():
    """检查YOLO模型文件"""
    print("\n📁 检查YOLO模型文件")
    print("-" * 40)
    
    model_paths = [
        "best1.pt",
        "runs/detect/train/weights/best1.pt",
        "yolov8n.pt",
        "yolov8s.pt",
        "../weights/best1.pt",
        "./weights/best1.pt"
    ]
    
    found_models = []
    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ 找到模型: {path} ({size_mb:.1f} MB)")
            found_models.append(path)
        else:
            print(f"❌ 未找到: {path}")
    
    return found_models

def test_ultralytics_export_jetson_optimized():
    """Jetson优化的TensorRT导出功能"""
    print("\n🎯 Jetson优化的YOLO TensorRT导出")
    print("-" * 40)
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics导入成功")
        
        # 查找可用的模型
        model_paths = test_yolo_model_exists()
        if not model_paths:
            print("❌ 没有找到YOLO模型文件")
            print("   请确保有best.pt或其他.pt模型文件")
            return False
        
        # 使用第一个找到的模型
        model_path = model_paths[0]
        print(f"使用模型: {model_path}")
        
        # 检查是否已有TensorRT引擎
        engine_path = model_path.replace('.pt', '_jetson.engine')
        if os.path.exists(engine_path):
            print(f"✅ 已存在TensorRT引擎: {engine_path}")
            return engine_path
        
        # 内存预检查
        if not check_memory_status():
            print("❌ 内存不足，无法进行TensorRT转换")
            return False
        
        # 释放内存
        free_memory()
        
        print("开始Jetson优化的TensorRT导出...")
        print("⏳ 这可能需要10-30分钟时间...")
        
        # 加载模型
        print("🔄 加载YOLO模型...")
        model = YOLO(model_path)
        
        # Jetson优化参数
        jetson_export_params = {
            'format': 'engine',
            'device': 0,  # 使用GPU
            'half': True,  # 使用FP16精度（Jetson友好）
            'workspace': 1,  # 1GB工作空间（Jetson Nano适用）
            'verbose': True,
            'batch': 1,  # 固定批次大小
            'simplify': True,  # 简化模型
        }
        
        # 根据可用内存调整参数
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 2:
            jetson_export_params['workspace'] = 0.5  # 减少工作空间
            print("⚠️  内存有限，使用较小的工作空间")
        elif available_gb > 4:
            jetson_export_params['workspace'] = 2  # 增加工作空间
        
        start_time = time.time()
        
        try:
            # 执行导出
            success = model.export(**jetson_export_params)
            
            export_time = time.time() - start_time
            
            if success:
                print(f"✅ TensorRT导出成功! 耗时: {export_time:.1f}秒")
                return engine_path
            else:
                print("❌ TensorRT导出失败")
                return False
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("❌ 内存不足导致导出失败")
                print("建议解决方案:")
                print("1. 增加Swap空间")
                print("2. 关闭其他程序释放内存")
                print("3. 使用更小的模型（如yolov8n.pt）")
                print("4. 减少workspace参数")
                return False
            else:
                raise e
            
    except Exception as e:
        print(f"❌ TensorRT导出测试失败: {e}")
        return False

def test_tensorrt_inference():
    """测试TensorRT推理"""
    print("\n⚡ 测试TensorRT推理性能")
    print("-" * 40)
    
    try:
        from ultralytics import YOLO
        import cv2
        
        # 查找TensorRT引擎
        engine_files = []
        for ext in ['*_jetson.engine', '*.engine']:
            engine_files.extend(Path('.').glob(ext))
        
        if not engine_files:
            print("❌ 没有找到TensorRT引擎文件")
            print("   请先运行导出测试")
            return False
        
        engine_path = str(engine_files[0])
        print(f"使用引擎: {engine_path}")
        
        # 加载TensorRT模型
        model = YOLO(engine_path)
        print("✅ TensorRT模型加载成功")
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 预热
        print("预热模型...")
        for _ in range(3):
            _ = model(test_img, verbose=False)
        
        # 性能测试
        print("开始性能测试...")
        times = []
        num_tests = 10
        
        for i in range(num_tests):
            start_time = time.time()
            results = model(test_img, verbose=False)
            inference_time = time.time() - start_time
            times.append(inference_time * 1000)  # 转换为毫秒
            print(f"推理 {i+1}/{num_tests}: {inference_time*1000:.1f}ms")
        
        # 统计结果
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time
        
        print(f"\n📊 性能统计:")
        print(f"平均推理时间: {avg_time:.1f}ms")
        print(f"最快推理时间: {min_time:.1f}ms")
        print(f"最慢推理时间: {max_time:.1f}ms")
        print(f"理论FPS: {fps:.1f}")
        
        # Jetson性能评估
        if avg_time < 50:  # 小于50ms
            print("✅ TensorRT推理性能优秀! (Jetson设备)")
        elif avg_time < 100:
            print("✅ TensorRT推理性能良好! (Jetson设备)")
        elif avg_time < 200:
            print("⚠️  TensorRT推理性能一般 (Jetson设备)")
        else:
            print("❌ TensorRT推理性能较差，可能需要优化")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorRT推理测试失败: {e}")
        return False

def compare_pytorch_vs_tensorrt():
    """对比PyTorch和TensorRT性能"""
    print("\n⚔️  PyTorch vs TensorRT 性能对比")
    print("-" * 50)
    
    try:
        from ultralytics import YOLO
        import cv2
        
        # 查找模型
        pt_models = [f for f in os.listdir('.') if f.endswith('.pt')]
        engine_models = [f for f in os.listdir('.') if f.endswith('.engine')]
        
        if not pt_models:
            print("❌ 没有找到PyTorch模型(.pt)")
            return False
        
        if not engine_models:
            print("❌ 没有找到TensorRT引擎(.engine)")
            return False
        
        pt_model = YOLO(pt_models[0])
        trt_model = YOLO(engine_models[0])
        
        # 测试图像
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 预热
        for model in [pt_model, trt_model]:
            for _ in range(3):
                _ = model(test_img, verbose=False)
        
        # 测试PyTorch
        print("测试PyTorch性能...")
        pt_times = []
        for _ in range(5):
            start = time.time()
            _ = pt_model(test_img, verbose=False)
            pt_times.append((time.time() - start) * 1000)
        
        # 测试TensorRT
        print("测试TensorRT性能...")
        trt_times = []
        for _ in range(5):
            start = time.time()
            _ = trt_model(test_img, verbose=False)
            trt_times.append((time.time() - start) * 1000)
        
        pt_avg = np.mean(pt_times)
        trt_avg = np.mean(trt_times)
        speedup = pt_avg / trt_avg
        
        print(f"\n📊 性能对比结果:")
        print(f"PyTorch平均时间: {pt_avg:.1f}ms ({1000/pt_avg:.1f} FPS)")
        print(f"TensorRT平均时间: {trt_avg:.1f}ms ({1000/trt_avg:.1f} FPS)")
        print(f"🚀 TensorRT加速倍数: {speedup:.1f}x")
        
        # Jetson设备加速评估
        if speedup > 3:
            print("✅ TensorRT加速效果优秀! (Jetson设备)")
        elif speedup > 2:
            print("✅ TensorRT加速效果良好! (Jetson设备)")
        elif speedup > 1.2:
            print("⚠️  TensorRT加速效果一般 (Jetson设备)")
        else:
            print("❌ TensorRT加速效果不明显")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能对比失败: {e}")
        return False

def jetson_optimization_tips():
    """Jetson优化建议"""
    print("\n💡 Jetson TensorRT优化建议")
    print("-" * 50)
    
    is_jetson, device_model = check_jetson_device()
    
    if is_jetson:
        print("🎯 针对Jetson设备的优化建议:")
        print("1. 设置最高性能模式:")
        print("   sudo nvpmodel -m 0")
        print("   sudo jetson_clocks")
        
        print("\n2. 增加Swap空间 (4-6GB):")
        print("   sudo fallocate -l 4G /swapfile")
        print("   sudo chmod 600 /swapfile") 
        print("   sudo mkswap /swapfile")
        print("   sudo swapon /swapfile")
        
        print("\n3. TensorRT导出优化参数:")
        print("   - 使用FP16精度 (half=True)")
        print("   - 减少工作空间 (workspace=1-2)")
        print("   - 固定批次大小 (batch=1)")
        print("   - 简化模型 (simplify=True)")
        
        print("\n4. 内存管理:")
        print("   - 关闭不必要的程序")
        print("   - 定期清理缓存")
        print("   - 监控内存使用 (htop/jtop)")
        
        print("\n5. 模型选择:")
        print("   - 优先使用轻量级模型 (YOLOv8n)")
        print("   - 避免过大的输入分辨率")
    else:
        print("⚠️  未检测到Jetson设备，通用优化建议:")
        print("1. 确保有足够的GPU内存")
        print("2. 使用合适的TensorRT版本")
        print("3. 选择合适的精度模式")

def main():
    """主函数"""
    print("🚀 Jetson TensorRT功能验证")
    print("=" * 50)
    
    # 检查设备类型
    is_jetson, device_model = check_jetson_device()
    
    # 测试步骤
    tests = [
        ("设备检查", lambda: is_jetson),
        ("内存状态", check_memory_status),
        ("Swap管理", setup_swap_if_needed),
        ("TensorRT导入", test_tensorrt_import),
        ("PyCUDA", test_pycuda),
        ("YOLO模型", lambda: len(test_yolo_model_exists()) > 0),
        ("TensorRT导出", test_ultralytics_export_jetson_optimized),
        ("TensorRT推理", test_tensorrt_inference),
        ("性能对比", compare_pytorch_vs_tensorrt)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = test_func()
            results[name] = result
            if result:
                print(f"✅ {name} 测试通过")
            else:
                print(f"❌ {name} 测试失败")
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}")
            results[name] = False
    
    # 显示优化建议
    jetson_optimization_tips()
    
    # 总结
    print(f"\n{'='*20} 测试总结 {'='*20}")
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过 ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！TensorRT完全可用！")
    elif passed >= total * 0.8:
        print("✅ 大部分测试通过，TensorRT基本可用")
    else:
        print("⚠️  多个测试失败，需要进一步排查问题")
    
    # 清理临时文件
    cleanup_swap()

if __name__ == "__main__":
    main() 