#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试箭头方向矫正
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from arrow_orientation_correction import ArrowOrientationCorrector

def save_user_image():
    """
    创建用户提供的图像（模拟保存）
    实际使用时，您需要将图像保存到指定位置
    """
    # 这里我们使用一个示例图像路径
    # 您需要将实际的图像保存为这个文件名
    return "DEMO_DETECT_TEST/user_arrow.jpg"

def test_arrow_correction():
    """测试箭头矫正功能"""
    
    print("🚀 开始箭头方向矫正测试")
    print("=" * 50)
    
    # 创建矫正器
    corrector = ArrowOrientationCorrector()
    
    # 测试图像路径
    test_image = save_user_image()
    
    # 检查图像是否存在
    if not os.path.exists(test_image):
        print(f"❌ 图像不存在: {test_image}")
        print("请将您的箭头图像保存为 'DEMO_DETECT_TEST/user_arrow.jpg'")
        
        # 或者测试现有的图像
        test_images = [
            "DEMO_DETECT_TEST/yolo_arrow_test_results/2_target_1_original.jpg",
            "DEMO_DETECT_TEST/yolo_arrow_test_results/1_target_1_original.jpg",
            "DEMO_DETECT_TEST/yolo_arrow_test_results/3_target_1_original.jpg"
        ]
        
        print("\n🔄 尝试使用现有测试图像...")
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"✅ 找到测试图像: {img_path}")
                test_image = img_path
                break
        else:
            print("❌ 未找到可用的测试图像")
            return
    
    # 执行矫正
    result = corrector.correct_arrow_orientation(test_image)
    
    # 显示结果
    if result["success"]:
        print("\n" + "="*50)
        print("🎉 箭头方向矫正成功！")
        print(f"📁 原始图像: {result['original_image']}")
        print(f"📁 矫正图像: {result['corrected_image']}")
        print(f"🎯 原始尖端角度: {result['tip_angle']:.1f}°")
        print(f"🔄 旋转角度: {result['rotation_angle']:.1f}°")
        print(f"📍 尖端位置: {result['tip_point']}")
        print(f"📏 轮廓面积: {result['contour_area']:.0f} 像素")
        
        # 显示矫正前后对比
        display_comparison(result['original_image'], result['corrected_image'])
        
    else:
        print(f"❌ 矫正失败: {result['error']}")

def display_comparison(original_path, corrected_path):
    """显示矫正前后对比"""
    
    if not (os.path.exists(original_path) and os.path.exists(corrected_path)):
        return
    
    # 读取图像
    original = cv2.imread(original_path)
    corrected = cv2.imread(corrected_path)
    
    if original is None or corrected is None:
        return
    
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("矫正前")
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[1].set_title("矫正后（箭头朝上）")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = "DEMO_DETECT_TEST/corrected_arrows/correction_comparison.jpg"
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 对比图保存至: {comparison_path}")

if __name__ == "__main__":
    test_arrow_correction() 