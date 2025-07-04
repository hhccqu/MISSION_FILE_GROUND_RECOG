#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双线程SITL系统测试脚本
用于验证系统各组件是否正常工作
"""

import os
import sys
import time
import cv2
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有必要的模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试基础模块
        import cv2
        import numpy as np
        import json
        import threading
        import queue
        print("  ✅ 基础模块导入成功")
    except ImportError as e:
        print(f"  ❌ 基础模块导入失败: {e}")
        return False
    
    try:
        # 测试YOLO相关
        from yolo_trt_utils import YOLOTRTDetector
        print("  ✅ YOLO检测器导入成功")
    except ImportError as e:
        print(f"  ❌ YOLO检测器导入失败: {e}")
        return False
    
    try:
        # 测试地理计算器
        from target_geo_calculator import FlightData, TargetGeoCalculator, TargetInfo
        print("  ✅ 地理坐标计算器导入成功")
    except ImportError as e:
        print(f"  ❌ 地理坐标计算器导入失败: {e}")
        return False
    
    try:
        # 测试OCR
        import easyocr
        print("  ✅ EasyOCR导入成功")
    except ImportError as e:
        print(f"  ❌ EasyOCR导入失败: {e}")
        return False
    
    try:
        # 测试MAVLink（可选）
        from pymavlink import mavutil
        print("  ✅ MAVLink导入成功")
    except ImportError as e:
        print(f"  ⚠️ MAVLink导入失败: {e} (将使用模拟模式)")
    
    return True

def test_model_file():
    """测试模型文件是否存在"""
    print("\n🔍 测试模型文件...")
    
    model_path = "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/weights/best1.pt"
    
    if os.path.exists(model_path):
        print(f"  ✅ 模型文件存在: {model_path}")
        file_size = os.path.getsize(model_path) / (1024*1024)  # MB
        print(f"  📊 文件大小: {file_size:.1f} MB")
        return True
    else:
        print(f"  ❌ 模型文件不存在: {model_path}")
        return False

def test_yolo_detector():
    """测试YOLO检测器"""
    print("\n🔍 测试YOLO检测器...")
    
    try:
        from yolo_trt_utils import YOLOTRTDetector
        
        model_path = "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/weights/best1.pt"
        if not os.path.exists(model_path):
            print("  ❌ 模型文件不存在，跳过检测器测试")
            return False
        
        detector = YOLOTRTDetector(model_path=model_path, conf_thres=0.25)
        print("  ✅ YOLO检测器创建成功")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 测试检测
        detections = detector.detect(test_image)
        print(f"  ✅ 检测测试完成，检测到 {len(detections)} 个目标")
        
        return True
        
    except Exception as e:
        print(f"  ❌ YOLO检测器测试失败: {e}")
        return False

def test_image_corrector():
    """测试图像转正器"""
    print("\n🔍 测试图像转正器...")
    
    try:
        from dual_thread_sitl_mission import ImageOrientationCorrector
        
        corrector = ImageOrientationCorrector(debug_mode=False)
        print("  ✅ 图像转正器创建成功")
        
        # 创建测试图像（红色三角形）
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # 绘制红色三角形
        points = np.array([[100, 50], [50, 150], [150, 150]], np.int32)
        cv2.fillPoly(test_image, [points], (0, 0, 255))
        
        # 测试转正
        corrected_image, info = corrector.correct_orientation(test_image)
        
        if info['success']:
            print(f"  ✅ 图像转正成功，旋转角度: {info['rotation_angle']:.1f}°")
        else:
            print(f"  ⚠️ 图像转正失败: {info['error_message']}")
        
        stats = corrector.get_stats()
        print(f"  📊 转正统计: 总数={stats['total_processed']}, 成功={stats['successful_corrections']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 图像转正器测试失败: {e}")
        return False

def test_geo_calculator():
    """测试地理坐标计算器"""
    print("\n🔍 测试地理坐标计算器...")
    
    try:
        from target_geo_calculator import FlightData, TargetGeoCalculator
        
        calculator = TargetGeoCalculator(camera_fov_h=60.0, camera_fov_v=45.0)
        calculator.image_width = 1920
        calculator.image_height = 1080
        print("  ✅ 地理坐标计算器创建成功")
        
        # 创建测试飞行数据
        flight_data = FlightData(
            timestamp=time.time(),
            latitude=39.7392,
            longitude=116.4074,
            altitude=100.0,
            pitch=0.0,
            roll=0.0,
            yaw=45.0,
            ground_speed=15.0,
            heading=45.0
        )
        
        # 测试坐标计算
        target_lat, target_lon = calculator.calculate_target_position(960, 540, flight_data)
        
        print(f"  ✅ 坐标计算成功: ({target_lat:.6f}, {target_lon:.6f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 地理坐标计算器测试失败: {e}")
        return False

def test_video_source():
    """测试视频源"""
    print("\n🔍 测试视频源...")
    
    video_path = "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/video2.mp4"
    
    if not os.path.exists(video_path):
        print(f"  ❌ 视频文件不存在: {video_path}")
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("  ❌ 无法打开视频文件")
            return False
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  ✅ 视频文件正常")
        print(f"  📊 视频信息: {width}x{height}, {fps:.1f}fps, {frame_count}帧")
        
        # 读取第一帧测试
        ret, frame = cap.read()
        if ret:
            print("  ✅ 视频帧读取正常")
        else:
            print("  ❌ 视频帧读取失败")
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"  ❌ 视频源测试失败: {e}")
        return False

def test_threading():
    """测试线程功能"""
    print("\n🔍 测试线程功能...")
    
    try:
        import threading
        import queue
        import time
        
        # 创建队列
        test_queue = queue.Queue(maxsize=10)
        results = []
        
        def producer():
            """生产者线程"""
            for i in range(5):
                test_queue.put(f"data_{i}")
                time.sleep(0.1)
        
        def consumer():
            """消费者线程"""
            while True:
                try:
                    data = test_queue.get(timeout=1)
                    results.append(data)
                    test_queue.task_done()
                except queue.Empty:
                    break
        
        # 启动线程
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join()
        consumer_thread.join()
        
        print(f"  ✅ 线程通信测试成功，处理了 {len(results)} 个数据包")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 线程功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 双线程SITL系统组件测试")
    print("=" * 50)
    
    test_results = []
    
    # 执行各项测试
    test_results.append(("模块导入", test_imports()))
    test_results.append(("模型文件", test_model_file()))
    test_results.append(("YOLO检测器", test_yolo_detector()))
    test_results.append(("图像转正器", test_image_corrector()))
    test_results.append(("地理计算器", test_geo_calculator()))
    test_results.append(("视频源", test_video_source()))
    test_results.append(("线程功能", test_threading()))
    
    # 汇总测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 测试完成: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常运行。")
        print("\n📋 下一步:")
        print("  1. 运行 python dual_thread_sitl_mission.py")
        print("  2. 或者启动SITL仿真器后再运行")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关组件。")
        return False

if __name__ == "__main__":
    main() 