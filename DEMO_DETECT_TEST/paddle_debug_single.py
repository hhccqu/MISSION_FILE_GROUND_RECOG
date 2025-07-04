#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 单张图片调试测试
"""

import os
import cv2
from paddleocr import PaddleOCR
import json

def debug_single_image():
    """调试单张图片的PaddleOCR识别"""
    
    print("🔍 开始单张图片调试测试")
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
    
    # 测试特定图片
    image_path = "DEMO_DETECT_TEST/yolo_arrow_test_results/2_target_1_original.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return
        
    print(f"\n📸 测试图片: {os.path.basename(image_path)}")
    
    # 检查图片基本信息
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 无法读取图片")
        return
        
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
                print(f"  结果 {j}: 类型 {type(result)}")
                
                # 检查OCRResult对象的所有属性
                print(f"    对象属性: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                
                # 尝试以字典方式访问
                try:
                    print(f"    字典键: {list(result.keys())}")
                    for key, value in result.items():
                        if key in ['rec_texts', 'rec_scores', 'rec_polys', 'dt_polys']:
                            print(f"    {key}: {value} (类型: {type(value)})")
                        elif key == 'input_path':
                            print(f"    {key}: {value}")
                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                            print(f"    {key}: 长度={len(value)}, 类型={type(value)}")
                            if len(value) <= 3:  # 只显示前几个元素
                                print(f"      内容: {value}")
                        else:
                            print(f"    {key}: {type(value)}")
                except Exception as e:
                    print(f"    字典访问失败: {e}")
                
                # 检查是否有识别到的文本
                if 'rec_texts' in result and result['rec_texts']:
                    texts = result['rec_texts']
                    scores = result.get('rec_scores', [])
                    polys = result.get('rec_polys', [])
                    
                    print(f"\n🔧 找到识别结果:")
                    print(f"    文本: {texts}")
                    print(f"    置信度: {scores}")
                    print(f"    多边形: {len(polys)} 个")
                    
                    # 测试数字筛选
                    digit_results = []
                    for i, text in enumerate(texts):
                        score = scores[i] if i < len(scores) else 0.0
                        poly = polys[i] if i < len(polys) else []
                        
                        print(f"    检查文本 {i}: '{text}'")
                        if any(c.isdigit() for c in text):
                            digits = ''.join(c for c in text if c.isdigit())
                            print(f"      提取数字: '{digits}'")
                            if len(digits) >= 1:
                                digit_results.append({
                                    'text': text,
                                    'digits': digits,
                                    'confidence': score,
                                    'bbox': poly
                                })
                                print(f"      ✅ 添加到数字结果")
                    
                    print(f"    最终数字结果: {digit_results}")
                else:
                    print("    ❌ 没有找到rec_texts或为空")
                
        else:
            print("❌ 没有识别结果")
                
    except Exception as e:
        print(f"❌ 识别过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🔍 调试测试完成")

if __name__ == "__main__":
    debug_single_image() 