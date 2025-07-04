#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
展示改进后的箭头方向矫正结果
重点展示顶点位于最高位置的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def display_improved_results():
    """展示改进后的矫正结果"""
    
    print("🎨 展示改进后的箭头方向矫正结果")
    print("🎯 重点：确保箭头顶点位于图像最高位置")
    print("=" * 60)
    
    result_dir = "test_improved_results"
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
    
    # 显示所有处理过程
    n_images = len(process_images)
    fig, axes = plt.subplots(n_images, 1, figsize=(16, 6*n_images))
    
    if n_images == 1:
        axes = [axes]
    
    for i, process_path in enumerate(process_images):
        img = cv2.imread(process_path)
        if img is not None:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            filename = os.path.basename(process_path)
            axes[i].set_title(f"矫正过程 - {filename}", fontsize=12, pad=10)
            axes[i].axis('off')
            
            print(f"✅ 显示: {filename}")
    
    plt.suptitle("改进后的箭头方向矫正 - 顶点位于最高位置", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_before_after_comparison():
    """创建前后对比图"""
    
    print("\n🔄 创建前后对比图...")
    
    # 查找原始图像和矫正图像
    original_dir = "yolo_arrow_test_results"
    result_dir = "test_improved_results"
    
    # 找到成功矫正的案例
    corrected_images = glob.glob(os.path.join(result_dir, "*_corrected.jpg"))
    
    if not corrected_images:
        print("❌ 未找到矫正后的图像")
        return
    
    # 创建对比图
    n_pairs = min(3, len(corrected_images))  # 最多显示3对
    fig, axes = plt.subplots(2, n_pairs, figsize=(n_pairs*5, 8))
    
    if n_pairs == 1:
        axes = axes.reshape(2, 1)
    
    for i, corrected_path in enumerate(corrected_images[:n_pairs]):
        # 构造原始图像路径
        base_name = os.path.basename(corrected_path).replace("_corrected.jpg", ".jpg")
        original_path = os.path.join(original_dir, base_name)
        
        if os.path.exists(original_path):
            # 读取图像
            original_img = cv2.imread(original_path)
            corrected_img = cv2.imread(corrected_path)
            
            if original_img is not None and corrected_img is not None:
                # 显示原始图像
                axes[0, i].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                axes[0, i].set_title(f"原始图像\n{base_name}", fontsize=10)
                axes[0, i].axis('off')
                
                # 显示矫正图像  
                axes[1, i].imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(f"矫正后 (顶点朝上)\n{base_name}", fontsize=10)
                axes[1, i].axis('off')
                
                print(f"✅ 对比: {base_name}")
    
    plt.suptitle("箭头方向矫正前后对比 - 顶点位置矫正", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def analyze_correction_effectiveness():
    """分析矫正效果"""
    
    print("\n📊 分析矫正效果...")
    
    from arrow_orientation_correction import ArrowOrientationCorrector
    
    result_dir = "test_improved_results"
    corrected_images = glob.glob(os.path.join(result_dir, "*_corrected.jpg"))
    
    corrector = ArrowOrientationCorrector()
    
    analysis_results = []
    
    for corrected_path in corrected_images:
        img = cv2.imread(corrected_path)
        if img is None:
            continue
            
        # 检测箭头
        contours = corrector.detect_arrow_contours(img)
        if not contours:
            continue
            
        largest_contour = max(contours, key=cv2.contourArea)
        tip_point = corrector.find_arrow_tip(largest_contour)
        
        if tip_point is None:
            continue
            
        # 分析顶点位置
        contour_points = largest_contour.reshape(-1, 2)
        min_y = np.min(contour_points[:, 1])
        tip_y = tip_point[1]
        
        y_diff = abs(tip_y - min_y)
        is_at_top = y_diff <= 15
        
        analysis_results.append({
            'filename': os.path.basename(corrected_path),
            'tip_y': tip_y,
            'min_y': min_y,
            'y_diff': y_diff,
            'is_at_top': is_at_top
        })
        
        status = "✅ 正确" if is_at_top else "❌ 需调整"
        print(f"{status} {os.path.basename(corrected_path)}: 顶点y={tip_y}, 最高点y={min_y}, 差异={y_diff}")
    
    # 统计结果
    total = len(analysis_results)
    correct = sum(1 for r in analysis_results if r['is_at_top'])
    
    print(f"\n📈 矫正效果统计:")
    print(f"总数量: {total}")
    print(f"顶点位置正确: {correct}")
    print(f"准确率: {correct/total*100:.1f}%" if total > 0 else "准确率: 0%")
    
    return analysis_results

if __name__ == "__main__":
    # 显示处理过程
    display_improved_results()
    
    # 创建前后对比
    create_before_after_comparison()
    
    # 分析效果
    analyze_correction_effectiveness()
    
    print("\n" + "=" * 60)
    print("🎉 改进后的箭头方向矫正系统完成！")
    print("🎯 核心改进：确保箭头顶点位于图像最高位置")
    print("✨ 这样的矫正更符合实际需求，有助于提高OCR识别准确率")
    print("=" * 60) 