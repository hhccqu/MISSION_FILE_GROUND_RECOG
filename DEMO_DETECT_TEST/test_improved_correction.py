#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的箭头方向矫正功能
确保箭头顶点位于图像最高位置
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from arrow_orientation_correction import ArrowOrientationCorrector

def test_single_image(image_path):
    """测试单张图像的矫正效果"""
    
    print(f"🧪 测试图像: {os.path.basename(image_path)}")
    print("=" * 50)
    
    # 创建矫正器
    corrector = ArrowOrientationCorrector()
    
    # 执行矫正
    result = corrector.correct_arrow_orientation(image_path, "test_improved_results")
    
    if result["success"]:
        print("✅ 矫正成功!")
        print(f"📍 原始顶点位置: {result['tip_point']}")
        print(f"📍 轮廓中心位置: {result['center_point']}")
        print(f"🔄 旋转角度: {result['rotation_angle']:.1f}°")
        print(f"📏 轮廓面积: {result['contour_area']:.0f} 像素")
        
        if result.get('already_correct', False):
            print("ℹ️ 箭头已经朝向正确，无需旋转")
        
        # 验证矫正结果
        if 'corrected_image' in result:
            verify_correction_result(result['corrected_image'])
        
    else:
        print(f"❌ 矫正失败: {result['error']}")
    
    print("\n" + "=" * 50)
    return result

def verify_correction_result(corrected_image_path):
    """验证矫正结果是否正确"""
    
    print("🔍 验证矫正结果...")
    
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
    
    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 检测顶点
    tip_point = corrector.find_arrow_tip(largest_contour)
    if tip_point is None:
        print("❌ 无法在矫正后图像中检测到箭头顶点")
        return False
    
    # 检查顶点是否位于最高位置
    contour_points = largest_contour.reshape(-1, 2)
    min_y = np.min(contour_points[:, 1])
    tip_y = tip_point[1]
    
    y_tolerance = 15
    is_at_top = abs(tip_y - min_y) <= y_tolerance
    
    print(f"📊 轮廓最高点y坐标: {min_y}")
    print(f"📊 顶点y坐标: {tip_y}")
    print(f"📊 y坐标差异: {abs(tip_y - min_y)}")
    
    if is_at_top:
        print("✅ 验证通过: 箭头顶点位于最高位置")
        return True
    else:
        print("❌ 验证失败: 箭头顶点未位于最高位置")
        return False

def test_multiple_images():
    """测试多张图像"""
    
    print("🚀 批量测试改进后的箭头矫正功能")
    print("=" * 60)
    
    # 查找测试图像
    test_images = []
    
    # 从YOLO结果中选择几张代表性图像
    yolo_dir = "yolo_arrow_test_results"
    if os.path.exists(yolo_dir):
        import glob
        pattern = os.path.join(yolo_dir, "*_original.jpg")
        yolo_images = glob.glob(pattern)
        test_images.extend(yolo_images[:5])  # 取前5张
    
    if not test_images:
        print("❌ 未找到测试图像")
        return
    
    # 创建输出目录
    os.makedirs("test_improved_results", exist_ok=True)
    
    # 统计结果
    total_tests = len(test_images)
    successful_corrections = 0
    successful_verifications = 0
    
    # 逐个测试
    for i, image_path in enumerate(test_images, 1):
        print(f"\n📸 测试 {i}/{total_tests}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        result = test_single_image(image_path)
        
        if result["success"]:
            successful_corrections += 1
            
            # 验证结果
            if 'corrected_image' in result:
                if verify_correction_result(result['corrected_image']):
                    successful_verifications += 1
    
    # 显示总体结果
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print(f"总测试数量: {total_tests}")
    print(f"矫正成功: {successful_corrections}")
    print(f"验证通过: {successful_verifications}")
    print(f"矫正成功率: {successful_corrections/total_tests*100:.1f}%")
    print(f"验证通过率: {successful_verifications/total_tests*100:.1f}%")
    
    if successful_verifications == total_tests:
        print("🎉 所有测试都通过了！箭头矫正功能工作正常")
    elif successful_verifications > total_tests * 0.8:
        print("✅ 大部分测试通过，箭头矫正功能基本正常")
    else:
        print("⚠️ 部分测试未通过，需要进一步调优")

def create_comparison_showcase():
    """创建对比展示"""
    
    result_dir = "test_improved_results"
    if not os.path.exists(result_dir):
        print("❌ 测试结果目录不存在")
        return
    
    # 查找处理过程图像
    import glob
    process_images = glob.glob(os.path.join(result_dir, "*_correction_process.jpg"))
    
    if not process_images:
        print("❌ 未找到处理过程图像")
        return
    
    # 显示前几个结果
    n_show = min(3, len(process_images))
    
    fig, axes = plt.subplots(n_show, 1, figsize=(15, 5*n_show))
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        img = cv2.imread(process_images[i])
        if img is not None:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"矫正过程 - {os.path.basename(process_images[i])}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("改进后的箭头方向矫正 - 顶点位于最高位置", fontsize=16, fontweight='bold')
    plt.show()

if __name__ == "__main__":
    # 执行测试
    test_multiple_images()
    
    # 创建展示
    try:
        create_comparison_showcase()
    except Exception as e:
        print(f"展示创建失败: {e}")
    
    print("\n🎯 测试完成！改进后的算法确保箭头顶点位于图像最高位置。") 