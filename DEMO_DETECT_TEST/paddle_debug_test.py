#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 调试测试 - 分析识别失败原因
"""

import os
import cv2
from paddleocr import PaddleOCR
import json

def debug_paddle_ocr():
    """调试PaddleOCR识别问题"""
    
    print("🔍 开始PaddleOCR调试测试")
    print("=" * 50)
    
    # 初始化PaddleOCR
    try:
        ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en'
        )
        print("✅ PaddleOCR初始化成功")
    except Exception as e:
        print(f"❌ PaddleOCR初始化失败: {e}")
        return
    
    # 测试几张图片
    test_images = [
        "DEMO_DETECT_TEST/yolo_arrow_test_results/1_target_1_original.jpg",
        "DEMO_DETECT_TEST/yolo_arrow_test_results/2_target_1_original.jpg",
        "DEMO_DETECT_TEST/yolo_arrow_test_results/3_target_1_original.jpg"
    ]
    
    for i, image_path in enumerate(test_images, 1):
        if not os.path.exists(image_path):
            print(f"❌ 图片不存在: {image_path}")
            continue
            
        print(f"\n📸 测试图片 {i}: {os.path.basename(image_path)}")
        
        # 检查图片基本信息
        img = cv2.imread(image_path)
        if img is None:
            print("❌ 无法读取图片")
            continue
            
        h, w = img.shape[:2]
        print(f"📏 图片尺寸: {w}x{h}")
        
        try:
            # 使用PaddleOCR识别
            results = ocr.predict(image_path)
            print(f"🔍 原始识别结果类型: {type(results)}")
            print(f"🔍 原始识别结果长度: {len(results) if results else 0}")
            
            if results:
                print("🔍 原始识别结果内容:")
                for j, result in enumerate(results):
                    print(f"  结果 {j}: {result}")
                    print(f"  结果类型: {type(result)}")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            print(f"    {key}: {value}")
                    elif isinstance(result, (list, tuple)):
                        for k, item in enumerate(result):
                            print(f"    项目 {k}: {item} (类型: {type(item)})")
            else:
                print("❌ 没有识别结果")
                
        except Exception as e:
            print(f"❌ 识别过程出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🔍 调试测试完成")

if __name__ == "__main__":
    debug_paddle_ocr() 