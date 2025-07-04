#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试基于tip-center连线垂直化的箭头方向矫正功能
新逻辑：将tip与center的连线转为垂直方向且tip在上方
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from arrow_orientation_correction import ArrowOrientationCorrector

def test_vertical_line_correction():
    """测试新的垂直连线矫正逻辑"""
    
    print("🚀 测试基于tip-center连线垂直化的箭头矫正")
    print("🎯 新逻辑：将tip与center的连线转为垂直方向且tip在上方")
    print("=" * 70)
    
    # 查找测试图像
    test_images = []
    
    # 从YOLO结果中选择几张代表性图像
    yolo_dir = "yolo_arrow_test_results"
    if os.path.exists(yolo_dir):
        pattern = os.path.join(yolo_dir, "*_original.jpg")
        yolo_images = glob.glob(pattern)
        test_images.extend(yolo_images[:5])  # 取前5张
    
    if not test_images:
        print("❌ 未找到测试图像")
        return
    
    # 创建输出目录
    output_dir = "vertical_line_correction_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建矫正器
    corrector = ArrowOrientationCorrector()
    
    # 统计结果
    total_tests = len(test_images)
    successful_corrections = 0
    successful_verifications = 0
    
    # 逐个测试
    for i, image_path in enumerate(test_images, 1):
        print(f"\n📸 测试 {i}/{total_tests}: {os.path.basename(image_path)}")
        print("-" * 50)
        
        # 执行矫正
        result = corrector.correct_arrow_orientation(image_path, output_dir)
        
        if result["success"]:
            successful_corrections += 1
            print("✅ 矫正成功!")
            print(f"📍 顶点位置: {result['tip_point']}")
            print(f"📍 中心位置: {result['center_point']}")
            print(f"🔄 旋转角度: {result['rotation_angle']:.1f}°")
            
            if result.get('already_correct', False):
                print("ℹ️ 连线已经垂直，无需旋转")
            
            # 验证结果
            if 'corrected_image' in result:
                if verify_vertical_line_correction(result['corrected_image']):
                    successful_verifications += 1
        else:
            print(f"❌ 矫正失败: {result['error']}")
    
    # 显示总体结果
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print(f"总测试数量: {total_tests}")
    print(f"矫正成功: {successful_corrections}")
    print(f"验证通过: {successful_verifications}")
    print(f"矫正成功率: {successful_corrections/total_tests*100:.1f}%")
    print(f"验证通过率: {successful_verifications/total_tests*100:.1f}%")
    
    if successful_verifications == total_tests:
        print("🎉 所有测试都通过了！新的垂直连线矫正逻辑工作正常")
    elif successful_verifications > total_tests * 0.8:
        print("✅ 大部分测试通过，新的矫正逻辑基本正常")
    else:
        print("⚠️ 部分测试未通过，需要进一步调优")

def verify_vertical_line_correction(corrected_image_path):
    """验证矫正结果是否符合垂直连线要求"""
    
    print("🔍 验证垂直连线矫正结果...")
    
    # 读取矫正后的图像
    corrected_img = cv2.imread(corrected_image_path)
    if corrected_img is None:
        print("❌ 无法读取矫正后的图像")
        return False
    
    # 重新检测箭头
    corrector = ArrowOrientationCorrector()
    contours = corrector.detect_arrow_contours(corrected_img)
    
    if not contours:
        print("❌ 矫正后图像中未检测到箭头轮廓")
        return False
    
    # 找到最大轮廓并计算中心
    largest_contour = max(contours, key=cv2.contourArea)
    
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        print("❌ 无法计算轮廓中心")
        return False
    
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    center_point = (center_x, center_y)
    
    # 检测顶点
    tip_point = corrector.find_arrow_tip(largest_contour)
    if tip_point is None:
        print("❌ 无法在矫正后图像中检测到箭头顶点")
        return False
    
    # 计算tip-center连线角度
    dx = tip_point[0] - center_point[0]
    dy = tip_point[1] - center_point[1]
    
    # 计算连线与垂直方向的夹角
    line_angle = np.degrees(np.arctan2(dx, -dy))
    
    # 检查条件
    angle_tolerance = 15  # 度
    is_vertical = abs(line_angle) <= angle_tolerance
    is_tip_above = dy < 0
    
    print(f"📊 中心点: {center_point}")
    print(f"📊 顶点: {tip_point}")
    print(f"📊 连线角度: {line_angle:.1f}° (相对垂直)")
    print(f"📊 是否垂直: {is_vertical} (误差≤{angle_tolerance}°)")
    print(f"📊 Tip在上方: {is_tip_above}")
    
    is_correct = is_vertical and is_tip_above
    
    if is_correct:
        print("✅ 验证通过: tip-center连线垂直且tip在上方")
        return True
    else:
        print("❌ 验证失败: tip-center连线未达到垂直要求")
        return False

def display_vertical_line_results():
    """展示垂直连线矫正结果"""
    
    print("\n🎨 展示垂直连线矫正结果")
    print("=" * 50)
    
    result_dir = "vertical_line_correction_results"
    if not os.path.exists(result_dir):
        print(f"❌ 结果目录不存在: {result_dir}")
        return
    
    # 查找处理过程图像
    process_images = glob.glob(os.path.join(result_dir, "*_correction_process.jpg"))
    process_images.sort()
    
    if not process_images:
        print("❌ 未找到处理过程图像")
        return
    
    print(f"📸 找到 {len(process_images)} 个处理过程图像")
    
    # 显示前3个结果
    n_show = min(3, len(process_images))
    
    fig, axes = plt.subplots(n_show, 1, figsize=(16, 6*n_show))
    if n_show == 1:
        axes = [axes]
    
    for i, process_path in enumerate(process_images[:n_show]):
        img = cv2.imread(process_path)
        if img is not None:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            filename = os.path.basename(process_path)
            axes[i].set_title(f"垂直连线矫正 - {filename}", fontsize=12, pad=10)
            axes[i].axis('off')
            
            print(f"✅ 显示: {filename}")
    
    plt.suptitle("基于Tip-Center连线垂直化的箭头方向矫正", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 执行测试
    test_vertical_line_correction()
    
    # 展示结果
    try:
        display_vertical_line_results()
    except Exception as e:
        print(f"展示创建失败: {e}")
    
    print("\n🎯 测试完成！新的垂直连线矫正逻辑确保tip-center连线垂直且tip在上方。") 