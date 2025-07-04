#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试单个图像的垂直连线矫正过程
详细分析每个步骤，找出问题所在
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from arrow_orientation_correction import ArrowOrientationCorrector

def debug_single_image_correction(image_path):
    """详细调试单个图像的矫正过程"""
    
    print(f"🔍 调试图像: {os.path.basename(image_path)}")
    print("=" * 60)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 无法读取图像")
        return
    
    # 创建矫正器
    corrector = ArrowOrientationCorrector()
    
    # 步骤1：检测箭头轮廓
    print("📋 步骤1：检测箭头轮廓")
    contours = corrector.detect_arrow_contours(image)
    if not contours:
        print("❌ 未检测到箭头轮廓")
        return
    
    largest_contour = max(contours, key=cv2.contourArea)
    print(f"✅ 检测到轮廓，面积: {cv2.contourArea(largest_contour):.0f}")
    
    # 步骤2：计算中心点
    print("\n📋 步骤2：计算中心点")
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        print("❌ 无法计算中心点")
        return
    
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    center_point = (center_x, center_y)
    print(f"✅ 中心点: {center_point}")
    
    # 步骤3：检测顶点
    print("\n📋 步骤3：检测顶点")
    tip_point = corrector.find_arrow_tip(largest_contour)
    if tip_point is None:
        print("❌ 无法检测顶点")
        return
    print(f"✅ 顶点: {tip_point}")
    
    # 步骤4：分析当前连线角度
    print("\n📋 步骤4：分析当前连线角度")
    dx = tip_point[0] - center_point[0]
    dy = tip_point[1] - center_point[1]
    current_angle = np.degrees(np.arctan2(dx, -dy))
    
    print(f"📊 向量: dx={dx}, dy={dy}")
    print(f"📊 当前角度: {current_angle:.1f}° (相对垂直向上)")
    print(f"📊 Tip在Center {'上方' if dy < 0 else '下方'}")
    
    # 步骤5：计算旋转角度
    print("\n📋 步骤5：计算旋转角度")
    rotation_angle = corrector.calculate_rotation_angle(tip_point, center_point, image.shape)
    
    # 步骤6：执行旋转
    print(f"\n📋 步骤6：执行旋转 ({rotation_angle:.1f}°)")
    rotated_image, rotation_matrix = corrector.rotate_image(image, rotation_angle, center_point)
    print(f"✅ 旋转完成，新图像尺寸: {rotated_image.shape[:2]}")
    
    # 步骤7：验证旋转结果
    print("\n📋 步骤7：验证旋转结果")
    
    # 重新检测旋转后的箭头
    new_contours = corrector.detect_arrow_contours(rotated_image)
    if not new_contours:
        print("❌ 旋转后未检测到轮廓")
        return
    
    new_largest_contour = max(new_contours, key=cv2.contourArea)
    print(f"✅ 新轮廓面积: {cv2.contourArea(new_largest_contour):.0f}")
    
    # 计算新的中心点
    new_M = cv2.moments(new_largest_contour)
    if new_M["m00"] == 0:
        print("❌ 无法计算新中心点")
        return
    
    new_center_x = int(new_M["m10"] / new_M["m00"])
    new_center_y = int(new_M["m01"] / new_M["m00"])
    new_center = (new_center_x, new_center_y)
    print(f"✅ 新中心点: {new_center}")
    
    # 检测新的顶点
    new_tip = corrector.find_arrow_tip(new_largest_contour)
    if new_tip is None:
        print("❌ 无法检测新顶点")
        return
    print(f"✅ 新顶点: {new_tip}")
    
    # 计算新的连线角度
    new_dx = new_tip[0] - new_center[0]
    new_dy = new_tip[1] - new_center[1]
    new_angle = np.degrees(np.arctan2(new_dx, -new_dy))
    
    print(f"📊 新向量: dx={new_dx}, dy={new_dy}")
    print(f"📊 新角度: {new_angle:.1f}° (相对垂直向上)")
    print(f"📊 新Tip在Center {'上方' if new_dy < 0 else '下方'}")
    
    # 评估结果
    angle_tolerance = 15
    is_vertical = abs(new_angle) <= angle_tolerance
    is_tip_above = new_dy < 0
    is_success = is_vertical and is_tip_above
    
    print(f"\n📊 最终评估:")
    print(f"   角度是否垂直: {is_vertical} (误差≤{angle_tolerance}°)")
    print(f"   Tip是否在上方: {is_tip_above}")
    print(f"   整体成功: {'✅ 是' if is_success else '❌ 否'}")
    
    # 保存调试图像
    save_debug_images(image, rotated_image, largest_contour, tip_point, center_point,
                     new_largest_contour, new_tip, new_center, rotation_angle, 
                     os.path.basename(image_path))
    
    return is_success

def save_debug_images(original, rotated, orig_contour, orig_tip, orig_center,
                     new_contour, new_tip, new_center, rotation_angle, filename):
    """保存调试图像"""
    
    # 创建原始图像标注
    orig_annotated = original.copy()
    cv2.drawContours(orig_annotated, [orig_contour], -1, (0, 255, 0), 2)
    cv2.circle(orig_annotated, orig_tip, 8, (0, 0, 255), -1)
    cv2.circle(orig_annotated, orig_center, 6, (255, 0, 0), -1)
    cv2.arrowedLine(orig_annotated, orig_center, orig_tip, (255, 255, 0), 3)
    
    # 添加垂直参考线
    ref_length = 50
    ref_top = (orig_center[0], max(0, orig_center[1] - ref_length))
    cv2.line(orig_annotated, orig_center, ref_top, (0, 255, 0), 2)
    
    # 创建旋转后图像标注
    rotated_annotated = rotated.copy()
    cv2.drawContours(rotated_annotated, [new_contour], -1, (0, 255, 0), 2)
    cv2.circle(rotated_annotated, new_tip, 8, (0, 0, 255), -1)
    cv2.circle(rotated_annotated, new_center, 6, (255, 0, 0), -1)
    cv2.arrowedLine(rotated_annotated, new_center, new_tip, (255, 255, 0), 3)
    
    # 添加垂直参考线
    ref_top_new = (new_center[0], max(0, new_center[1] - ref_length))
    cv2.line(rotated_annotated, new_center, ref_top_new, (0, 255, 0), 2)
    
    # 保存图像
    debug_dir = "debug_single_correction"
    os.makedirs(debug_dir, exist_ok=True)
    
    base_name = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(debug_dir, f"{base_name}_original_debug.jpg"), orig_annotated)
    cv2.imwrite(os.path.join(debug_dir, f"{base_name}_rotated_debug.jpg"), rotated_annotated)
    
    print(f"💾 调试图像已保存到 {debug_dir} 目录")

def test_multiple_debug():
    """测试多个图像的调试"""
    
    print("🚀 批量调试垂直连线矫正")
    print("=" * 60)
    
    # 查找测试图像
    import glob
    yolo_dir = "yolo_arrow_test_results"
    if not os.path.exists(yolo_dir):
        print("❌ 未找到YOLO结果目录")
        return
    
    pattern = os.path.join(yolo_dir, "*_original.jpg")
    test_images = glob.glob(pattern)[:3]  # 取前3张
    
    if not test_images:
        print("❌ 未找到测试图像")
        return
    
    success_count = 0
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n🔍 调试 {i}/{len(test_images)}")
        success = debug_single_image_correction(image_path)
        if success:
            success_count += 1
        print("-" * 60)
    
    print(f"\n📊 调试总结:")
    print(f"成功率: {success_count}/{len(test_images)} ({success_count/len(test_images)*100:.1f}%)")

if __name__ == "__main__":
    test_multiple_debug() 