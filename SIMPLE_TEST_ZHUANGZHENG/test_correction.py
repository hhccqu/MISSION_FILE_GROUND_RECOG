#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试图像方向校正系统
处理ORIGINAL_PICS目录中的示例图像
"""

import os
import sys
from image_orientation_corrector import process_images_batch

def test_image_correction():
    """测试图像校正功能"""
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "ORIGINAL_PICS")
    output_dir = os.path.join(current_dir, "CORRECTED_PICS")
    
    print("=" * 60)
    print("图像方向自动校正测试")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 列出输入目录中的文件
    input_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not input_files:
        print("错误: 在输入目录中未找到图像文件")
        return
    
    print(f"找到 {len(input_files)} 个图像文件:")
    for i, filename in enumerate(sorted(input_files), 1):
        print(f"  {i}. {filename}")
    print()
    
    # 执行批量处理（开启调试模式）
    print("开始处理图像...")
    print("-" * 40)
    
    try:
        results = process_images_batch(input_dir, output_dir, debug_mode=True)
        
        print()
        print("=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"总文件数: {results['total']}")
        print(f"成功处理: {results['success']}")
        print(f"处理失败: {results['failed']}")
        print()
        
        # 显示详细结果
        print("详细结果:")
        print("-" * 40)
        for detail in results['details']:
            status = "✓ 成功" if detail['success'] else "✗ 失败"
            print(f"{detail['filename']}: {status}")
            if detail['success']:
                print(f"  旋转角度: {detail['rotation_angle']:.2f}度")
                print(f"  尖端点: {detail['tip_point']}")
                print(f"  轮廓面积: {detail['contour_area']}")
            else:
                print(f"  错误: {detail['error_message']}")
            print()
        
        print(f"校正后的图像保存在: {output_dir}")
        print(f"调试图像保存在: {os.path.join(output_dir, 'debug')}")
        print(f"处理报告保存在: {os.path.join(output_dir, 'processing_report.txt')}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_image_correction() 