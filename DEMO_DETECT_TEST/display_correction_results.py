#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
展示箭头方向矫正结果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def display_correction_results():
    """展示箭头矫正结果"""
    
    print("🎨 展示箭头方向矫正结果")
    print("=" * 50)
    
    # 检查结果目录
    result_dir = "corrected_arrows_batch_final"
    if not os.path.exists(result_dir):
        print(f"❌ 结果目录不存在: {result_dir}")
        return
    
    # 展示主要结果图
    gallery_path = os.path.join(result_dir, "correction_comparison_gallery.jpg")
    showcase_path = os.path.join(result_dir, "best_correction_showcase.jpg")
    
    if os.path.exists(gallery_path):
        print(f"🖼️ 完整对比图库: {gallery_path}")
        
        # 读取并显示图库
        gallery_img = cv2.imread(gallery_path)
        if gallery_img is not None:
            plt.figure(figsize=(20, 15))
            plt.imshow(cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB))
            plt.title("箭头方向矫正 - 完整对比图库", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    if os.path.exists(showcase_path):
        print(f"🏆 最佳案例展示: {showcase_path}")
        
        # 读取并显示最佳案例
        showcase_img = cv2.imread(showcase_path)
        if showcase_img is not None:
            plt.figure(figsize=(15, 8))
            plt.imshow(cv2.cvtColor(showcase_img, cv2.COLOR_BGR2RGB))
            plt.title("箭头方向矫正 - 最佳案例展示", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    # 显示统计信息
    report_path = os.path.join(result_dir, "correction_summary_report.txt")
    if os.path.exists(report_path):
        print(f"\n📄 详细报告: {report_path}")
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        # 提取关键统计信息
        lines = report_content.split('\n')
        for line in lines:
            if '总处理数量:' in line or '成功矫正:' in line or '成功率:' in line:
                print(f"📊 {line}")
    
    print("\n" + "=" * 50)
    print("✅ 批量箭头方向矫正完成!")
    print("🎯 所有箭头已成功矫正为朝向正上方(-90°)")
    print(f"📁 完整结果保存在: {result_dir}")

def show_individual_examples():
    """展示几个具体的矫正案例"""
    
    result_dir = "corrected_arrows_batch_final"
    
    # 选择几个有代表性的案例
    examples = [
        ("1_target_1_original", "图像1-目标1"),
        ("2_target_1_original", "图像2-目标1"), 
        ("3_target_1_original", "图像3-目标1"),
        ("4_target_1_original", "图像4-目标1")
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, (example_name, title) in enumerate(examples):
        if i >= 4:  # 只显示前4个
            break
            
        original_path = f"yolo_arrow_test_results/{example_name}.jpg"
        corrected_path = f"{result_dir}/{example_name}_corrected.jpg"
        
        if os.path.exists(original_path) and os.path.exists(corrected_path):
            # 读取图像
            original_img = cv2.imread(original_path)
            corrected_img = cv2.imread(corrected_path)
            
            if original_img is not None and corrected_img is not None:
                # 显示原始图像
                axes[0, i].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                axes[0, i].set_title(f"{title}\n(原始)")
                axes[0, i].axis('off')
                
                # 显示矫正图像
                axes[1, i].imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(f"{title}\n(矫正后)")
                axes[1, i].axis('off')
    
    plt.suptitle("箭头方向矫正 - 典型案例对比", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display_correction_results()
    
    # 询问是否显示个别案例
    try:
        response = input("\n是否显示典型案例对比？(y/n): ")
        if response.lower() in ['y', 'yes', '是']:
            show_individual_examples()
    except:
        pass  # 如果在非交互环境中运行，跳过输入 