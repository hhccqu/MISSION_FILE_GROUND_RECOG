#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理YOLO测试结果中的original图像进行箭头方向矫正
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from arrow_orientation_correction import ArrowOrientationCorrector

def find_original_images():
    """查找所有的original图像"""
    
    base_dir = "yolo_arrow_test_results"  # 修正路径
    pattern = os.path.join(base_dir, "*_original.jpg")
    
    original_images = glob.glob(pattern)
    original_images.sort()  # 排序以便有序处理
    
    print(f"🔍 找到 {len(original_images)} 张原始图像:")
    for img in original_images:
        print(f"  📸 {os.path.basename(img)}")
    
    return original_images

def batch_correct_arrows():
    """批量矫正箭头方向"""
    
    print("🚀 开始批量箭头方向矫正")
    print("=" * 60)
    
    # 创建矫正器
    corrector = ArrowOrientationCorrector()
    
    # 查找所有original图像
    original_images = find_original_images()
    
    if not original_images:
        print("❌ 未找到任何original图像")
        return
    
    # 创建输出目录
    output_dir = "DEMO_DETECT_TEST/corrected_arrows_batch"
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计变量
    total_images = len(original_images)
    successful_corrections = 0
    failed_corrections = 0
    results = []
    
    # 处理每张图像
    for i, image_path in enumerate(original_images, 1):
        print(f"\n📸 处理图像 {i}/{total_images}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        # 执行矫正
        result = corrector.correct_arrow_orientation(image_path, output_dir)
        
        # 记录结果
        result['image_name'] = os.path.basename(image_path)
        results.append(result)
        
        if result["success"]:
            successful_corrections += 1
            print(f"✅ 矫正成功！")
            print(f"   🎯 原始角度: {result['tip_angle']:.1f}°")
            print(f"   🔄 旋转角度: {result['rotation_angle']:.1f}°")
            print(f"   📏 轮廓面积: {result['contour_area']:.0f} 像素")
            print(f"   💾 保存位置: {os.path.basename(result['corrected_image'])}")
        else:
            failed_corrections += 1
            print(f"❌ 矫正失败: {result['error']}")
    
    # 生成汇总报告
    generate_summary_report(results, output_dir, successful_corrections, failed_corrections, total_images)
    
    # 创建对比展示
    create_comparison_gallery(results, output_dir)
    
    print("\n" + "=" * 60)
    print("📊 批量处理完成！")
    print(f"总处理数量: {total_images}")
    print(f"成功矫正: {successful_corrections}")
    print(f"失败数量: {failed_corrections}")
    print(f"成功率: {successful_corrections/total_images*100:.1f}%")
    print(f"📁 所有结果保存在: {output_dir}")

def generate_summary_report(results, output_dir, successful, failed, total):
    """生成汇总报告"""
    
    report_path = os.path.join(output_dir, "correction_summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("箭头方向矫正批量处理报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("📊 处理统计\n")
        f.write("-" * 30 + "\n")
        f.write(f"总处理数量: {total}\n")
        f.write(f"成功矫正: {successful}\n")
        f.write(f"失败数量: {failed}\n")
        f.write(f"成功率: {successful/total*100:.1f}%\n\n")
        
        f.write("📋 详细结果\n")
        f.write("-" * 30 + "\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['image_name']}\n")
            if result['success']:
                f.write(f"   ✅ 成功 - 原始角度: {result['tip_angle']:.1f}°, 旋转: {result['rotation_angle']:.1f}°\n")
                f.write(f"   📏 轮廓面积: {result['contour_area']:.0f} 像素\n")
            else:
                f.write(f"   ❌ 失败 - {result['error']}\n")
            f.write("\n")
    
    print(f"📄 汇总报告保存至: {report_path}")

def create_comparison_gallery(results, output_dir):
    """创建对比展示图库"""
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ 没有成功的矫正结果，无法创建对比图库")
        return
    
    # 计算网格布局
    n_images = len(successful_results)
    n_cols = min(4, n_images)  # 最多4列
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # 创建大图展示
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 4, n_rows * 6))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif n_rows == 1:
        axes = axes.reshape(2, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.reshape(n_rows * 2, n_cols)
    
    for i, result in enumerate(successful_results):
        row = (i // n_cols) * 2
        col = i % n_cols
        
        # 读取原始图像和矫正图像
        original_img = cv2.imread(result['original_image'])
        corrected_img = cv2.imread(result['corrected_image'])
        
        if original_img is not None and corrected_img is not None:
            # 显示原始图像
            axes[row, col].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f"原始: {result['image_name']}")
            axes[row, col].axis('off')
            
            # 显示矫正图像
            axes[row + 1, col].imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
            axes[row + 1, col].set_title(f"矫正后 (旋转{result['rotation_angle']:.1f}°)")
            axes[row + 1, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(successful_results), n_rows * n_cols):
        row = (i // n_cols) * 2
        col = i % n_cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')
    
    plt.tight_layout()
    
    # 保存对比图库
    gallery_path = os.path.join(output_dir, "correction_comparison_gallery.jpg")
    plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ 对比图库保存至: {gallery_path}")
    
    # 创建单独的成功案例展示
    create_success_showcase(successful_results, output_dir)

def create_success_showcase(successful_results, output_dir):
    """创建成功案例展示"""
    
    if len(successful_results) < 3:
        return
    
    # 选择前3个最好的结果（基于轮廓面积）
    best_results = sorted(successful_results, key=lambda x: x['contour_area'], reverse=True)[:3]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, result in enumerate(best_results):
        # 读取图像
        original_img = cv2.imread(result['original_image'])
        corrected_img = cv2.imread(result['corrected_image'])
        
        if original_img is not None and corrected_img is not None:
            # 原始图像
            axes[0, i].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f"原始: {result['image_name']}\n角度: {result['tip_angle']:.1f}°")
            axes[0, i].axis('off')
            
            # 矫正图像
            axes[1, i].imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f"矫正后 (箭头朝上)\n旋转: {result['rotation_angle']:.1f}°")
            axes[1, i].axis('off')
    
    plt.suptitle("箭头方向矫正 - 最佳案例展示", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存展示图
    showcase_path = os.path.join(output_dir, "best_correction_showcase.jpg")
    plt.savefig(showcase_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🏆 最佳案例展示保存至: {showcase_path}")

if __name__ == "__main__":
    batch_correct_arrows() 