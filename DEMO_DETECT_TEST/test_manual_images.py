#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动测试图像的箭头方向修正测试
使用用户提供的测试图像验证算法效果
"""

import cv2
import numpy as np
import os
import time
from arrow_orientation_fix import ArrowOrientationFixer

def test_manual_images():
    """测试手动选择的图像"""
    print("🧪 开始测试手动选择的图像...")
    
    # 初始化箭头方向修正器
    fixer = ArrowOrientationFixer()
    
    # 测试图像目录
    test_dir = "test_image_manuel"
    
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        return
    
    # 获取所有PNG图像
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.png')]
    image_files.sort()  # 按文件名排序
    
    print(f"📁 找到 {len(image_files)} 个测试图像")
    
    # 创建结果目录
    results_dir = "manual_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 测试结果统计
    total_tests = 0
    successful_corrections = 0
    ocr_improvements = 0
    
    print("\n" + "="*80)
    
    for i, filename in enumerate(image_files, 1):
        print(f"\n🖼️  测试图像 {i}/{len(image_files)}: {filename}")
        print("-" * 60)
        
        filepath = os.path.join(test_dir, filename)
        image = cv2.imread(filepath)
        
        if image is None:
            print(f"❌ 无法加载图像: {filename}")
            continue
        
        total_tests += 1
        base_name = os.path.splitext(filename)[0]
        
        # 保存原始图像
        original_path = os.path.join(results_dir, f"{base_name}_original.jpg")
        cv2.imwrite(original_path, image)
        
        print(f"📏 图像尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 方法1: 箭头方向检测和修正
        print("\n🧭 方法1: 箭头方向检测和修正")
        start_time = time.time()
        
        corrected_image, was_corrected = fixer.correct_arrow_orientation(image)
        
        correction_time = time.time() - start_time
        print(f"⏱️  处理时间: {correction_time:.2f}秒")
        
        if was_corrected:
            successful_corrections += 1
            corrected_path = os.path.join(results_dir, f"{base_name}_corrected.jpg")
            cv2.imwrite(corrected_path, corrected_image)
            print(f"✅ 已修正并保存: {corrected_path}")
        else:
            print("ℹ️  无需修正或检测失败")
        
        # 方法2: 智能旋转与OCR验证
        print("\n🎯 方法2: 智能旋转与OCR验证")
        start_time = time.time()
        
        smart_image, ocr_text, ocr_confidence = fixer.smart_rotate_with_ocr_validation(image)
        
        smart_time = time.time() - start_time
        print(f"⏱️  处理时间: {smart_time:.2f}秒")
        
        smart_path = os.path.join(results_dir, f"{base_name}_smart.jpg")
        cv2.imwrite(smart_path, smart_image)
        print(f"💾 智能旋转结果: {smart_path}")
        
        if ocr_text:
            print(f"📝 OCR识别: '{ocr_text}' (置信度: {ocr_confidence:.2f})")
            if ocr_confidence > 0.5:
                ocr_improvements += 1
        else:
            print("📝 OCR识别: 无结果")
        
        # 对比测试：原始图像的OCR结果
        print("\n📊 对比测试: 原始图像OCR")
        try:
            original_results = fixer.ocr_reader.readtext(image)
            if original_results:
                best_original = max(original_results, key=lambda x: x[2])
                orig_text = best_original[1]
                orig_confidence = best_original[2]
                print(f"📝 原始OCR: '{orig_text}' (置信度: {orig_confidence:.2f})")
                
                # 比较改进效果
                if ocr_confidence > orig_confidence:
                    improvement = ocr_confidence - orig_confidence
                    print(f"📈 置信度提升: +{improvement:.2f}")
                elif ocr_confidence < orig_confidence:
                    decline = orig_confidence - ocr_confidence
                    print(f"📉 置信度下降: -{decline:.2f}")
                else:
                    print("📊 置信度无变化")
            else:
                print("📝 原始OCR: 无结果")
        except Exception as e:
            print(f"⚠️  原始OCR测试失败: {e}")
        
        print("=" * 60)
    
    # 输出测试摘要
    print(f"\n📋 测试摘要")
    print("=" * 80)
    print(f"🖼️  总测试图像: {total_tests}")
    print(f"🔄 成功修正: {successful_corrections} ({successful_corrections/max(1,total_tests)*100:.1f}%)")
    print(f"📈 OCR改进: {ocr_improvements} ({ocr_improvements/max(1,total_tests)*100:.1f}%)")
    print(f"📁 结果保存在: {results_dir}/")
    
    # 创建测试报告
    create_test_report(results_dir, total_tests, successful_corrections, ocr_improvements)

def create_test_report(results_dir: str, total: int, corrections: int, improvements: int):
    """创建测试报告"""
    report_path = os.path.join(results_dir, "test_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("箭头方向修正算法测试报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试图像数量: {total}\n")
        f.write(f"成功修正数量: {corrections}\n")
        f.write(f"OCR改进数量: {improvements}\n")
        f.write(f"修正成功率: {corrections/max(1,total)*100:.1f}%\n")
        f.write(f"OCR改进率: {improvements/max(1,total)*100:.1f}%\n\n")
        
        f.write("算法说明:\n")
        f.write("1. 箭头方向检测: 基于HSV色彩空间和凸包缺陷分析\n")
        f.write("2. 智能旋转: 四方向测试结合OCR验证\n")
        f.write("3. 质量保证: 高质量仿射变换和边界自适应\n")
    
    print(f"📄 测试报告已保存: {report_path}")

if __name__ == "__main__":
    test_manual_images() 