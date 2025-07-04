#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试SITL系统集成图像转正功能
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../SIMPLE_TEST_ZHUANGZHENG")

# 导入我们的模块
from sitl_strike_mission import ImageOrientationCorrector

def test_orientation_corrector():
    """测试图像转正器"""
    print("🧪 测试高精度图像转正器")
    print("=" * 40)
    
    # 创建转正器
    corrector = ImageOrientationCorrector(debug_mode=True)
    
    # 测试图像路径
    test_images_dir = "../SIMPLE_TEST_ZHUANGZHENG/ORIGINAL_PICS"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ 测试图像目录不存在: {test_images_dir}")
        return
    
    # 获取测试图像
    test_images = list(Path(test_images_dir).glob("*.png"))
    
    if not test_images:
        print(f"❌ 未找到测试图像")
        return
    
    print(f"📁 找到 {len(test_images)} 个测试图像")
    
    # 创建输出目录
    output_dir = "test_correction_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试每个图像
    for i, image_path in enumerate(test_images[:3]):  # 只测试前3个
        print(f"\n🖼️ 测试图像 {i+1}: {image_path.name}")
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            continue
        
        # 进行转正
        corrected_image, info = corrector.correct_orientation(image)
        
        # 显示结果
        if info['success']:
            print(f"  ✅ 转正成功")
            print(f"  🔄 旋转角度: {info['rotation_angle']:.2f}°")
            print(f"  📍 尖端点: {info['tip_point']}")
            print(f"  📐 轮廓面积: {info['contour_area']}")
        else:
            print(f"  ❌ 转正失败: {info['error_message']}")
        
        # 保存结果
        output_path = os.path.join(output_dir, f"corrected_{image_path.name}")
        cv2.imwrite(output_path, corrected_image)
        print(f"  💾 保存到: {output_path}")
    
    # 显示统计
    stats = corrector.get_stats()
    print(f"\n📊 转正统计:")
    print(f"  总处理: {stats['total_processed']}")
    print(f"  成功: {stats['successful_corrections']}")
    print(f"  失败: {stats['failed_corrections']}")
    if stats['total_processed'] > 0:
        success_rate = (stats['successful_corrections'] / stats['total_processed']) * 100
        print(f"  成功率: {success_rate:.1f}%")

def test_sitl_integration():
    """测试SITL系统集成"""
    print("\n🛩️ 测试SITL系统集成")
    print("=" * 40)
    
    try:
        from sitl_strike_mission import SITLStrikeMissionSystem
        
        # 创建配置
        config = {
            'conf_threshold': 0.25,
            'camera_fov_h': 60.0,
            'camera_fov_v': 45.0,
            'altitude': 100.0,
            'save_file': 'test_sitl_targets.json',
            'min_confidence': 0.5,
            'ocr_interval': 5,
            'max_targets_per_frame': 5,
            'orientation_correction': True,
            'correction_debug': False,
        }
        
        # 创建系统实例（不连接SITL）
        system = SITLStrikeMissionSystem(config, "test_connection")
        
        # 测试转正器初始化
        system.orientation_corrector = ImageOrientationCorrector(debug_mode=False)
        
        print("✅ SITL系统创建成功")
        print("✅ 图像转正器集成成功")
        
        # 测试转正方法
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[40:60, 40:60] = [0, 0, 255]  # 红色方块
        
        result = system._rotate_arrow(test_image)
        
        if result is not None:
            print("✅ 转正方法调用成功")
        else:
            print("❌ 转正方法调用失败")
        
        # 测试统计
        stats = system.orientation_corrector.get_stats()
        print(f"📊 转正统计: {stats}")
        
    except Exception as e:
        print(f"❌ SITL系统测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🧪 SITL图像转正集成测试")
    print("=" * 50)
    
    # 测试1: 转正器基础功能
    test_orientation_corrector()
    
    # 测试2: SITL系统集成
    test_sitl_integration()
    
    print("\n✅ 测试完成!")

if __name__ == "__main__":
    main() 