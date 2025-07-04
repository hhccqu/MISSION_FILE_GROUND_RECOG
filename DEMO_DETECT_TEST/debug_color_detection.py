#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试颜色检测和箭头识别
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def debug_color_detection(image_path):
    """调试颜色检测"""
    
    print(f"🔍 调试图像: {os.path.basename(image_path)}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 无法读取图像")
        return
    
    print(f"📏 图像尺寸: {image.shape}")
    
    # 转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 分析图像的颜色分布
    print("\n🎨 分析图像颜色分布...")
    
    # 计算每个像素的HSV值
    h_values = hsv[:,:,0].flatten()
    s_values = hsv[:,:,1].flatten()
    v_values = hsv[:,:,2].flatten()
    
    print(f"H (色调) 范围: {h_values.min()} - {h_values.max()}")
    print(f"S (饱和度) 范围: {s_values.min()} - {s_values.max()}")
    print(f"V (明度) 范围: {v_values.min()} - {v_values.max()}")
    
    # 尝试不同的颜色范围
    color_ranges = [
        ("红色1", np.array([0, 50, 50]), np.array([10, 255, 255])),
        ("红色2", np.array([170, 50, 50]), np.array([180, 255, 255])),
        ("粉色1", np.array([140, 50, 50]), np.array([170, 255, 255])),
        ("粉色2", np.array([150, 30, 100]), np.array([180, 255, 255])),
        ("宽泛粉色", np.array([120, 30, 50]), np.array([180, 255, 255])),
        ("全范围", np.array([0, 30, 50]), np.array([180, 255, 255]))
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # 显示原图
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    # 显示HSV图像
    axes[1].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    axes[1].set_title("HSV图像")
    axes[1].axis('off')
    
    best_mask = None
    best_contour_area = 0
    best_range_name = ""
    
    # 测试每个颜色范围
    for i, (name, lower, upper) in enumerate(color_ranges):
        mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_area = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            if contour_area > best_contour_area:
                best_contour_area = contour_area
                best_mask = mask.copy()
                best_range_name = name
        
        # 显示掩码
        if i + 2 < len(axes):
            axes[i + 2].imshow(mask, cmap='gray')
            axes[i + 2].set_title(f"{name}\n面积: {contour_area:.0f}")
            axes[i + 2].axis('off')
        
        print(f"  {name}: 最大轮廓面积 = {contour_area:.0f}")
    
    print(f"\n✅ 最佳颜色范围: {best_range_name} (面积: {best_contour_area:.0f})")
    
    plt.tight_layout()
    
    # 保存调试结果
    debug_path = f"DEMO_DETECT_TEST/debug_{os.path.basename(image_path)}_color.jpg"
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 调试结果保存至: {debug_path}")
    
    return best_mask, best_contour_area > 100  # 返回是否检测成功

def test_multiple_images():
    """测试多张图像"""
    
    test_images = [
        "DEMO_DETECT_TEST/user_arrow.jpg",
        "DEMO_DETECT_TEST/yolo_arrow_test_results/2_target_1_original.jpg",
        "DEMO_DETECT_TEST/yolo_arrow_test_results/1_target_1_original.jpg",
        "DEMO_DETECT_TEST/yolo_arrow_test_results/3_target_1_original.jpg"
    ]
    
    print("🚀 开始多图像颜色检测调试")
    print("=" * 50)
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n📸 测试图像: {image_path}")
            mask, success = debug_color_detection(image_path)
            if success:
                print("✅ 颜色检测成功")
            else:
                print("❌ 颜色检测失败")
        else:
            print(f"❌ 图像不存在: {image_path}")

if __name__ == "__main__":
    test_multiple_images() 