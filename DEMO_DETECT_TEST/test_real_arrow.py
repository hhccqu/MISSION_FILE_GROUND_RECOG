#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真实箭头图像的方向矫正
"""

import cv2
import numpy as np
from arrow_orientation_correction import ArrowOrientationCorrector
import os

def test_real_arrow():
    """测试真实的箭头图像"""
    
    print("🚀 开始真实箭头图像测试")
    print("=" * 50)
    
    # 创建矫正器
    corrector = ArrowOrientationCorrector()
    
    # 测试现有的图像
    test_images = [
        "DEMO_DETECT_TEST/yolo_arrow_test_results/2_target_1_original.jpg",
        "DEMO_DETECT_TEST/yolo_arrow_test_results/1_target_1_original.jpg", 
        "DEMO_DETECT_TEST/yolo_arrow_test_results/3_target_1_original.jpg"
    ]
    
    successful_corrections = 0
    total_tests = 0
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"❌ 图像不存在: {image_path}")
            continue
            
        total_tests += 1
        print(f"\n📸 测试图像 {total_tests}: {os.path.basename(image_path)}")
        
        # 执行矫正
        result = corrector.correct_arrow_orientation(image_path)
        
        if result["success"]:
            successful_corrections += 1
            print(f"✅ 矫正成功！")
            print(f"   原始角度: {result['tip_angle']:.1f}°")
            print(f"   需要旋转: {result['rotation_angle']:.1f}°")
            print(f"   轮廓面积: {result['contour_area']:.0f}")
            print(f"   矫正图像: {result['corrected_image']}")
        else:
            print(f"❌ 矫正失败: {result['error']}")
    
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print(f"总测试数量: {total_tests}")
    print(f"成功矫正: {successful_corrections}")
    print(f"成功率: {successful_corrections/total_tests*100:.1f}%" if total_tests > 0 else "0%")
    
    if successful_corrections > 0:
        print(f"\n📁 所有矫正结果保存在: DEMO_DETECT_TEST/corrected_arrows/")
        print("🎯 箭头已被矫正为朝向正上方")

if __name__ == "__main__":
    test_real_arrow() 