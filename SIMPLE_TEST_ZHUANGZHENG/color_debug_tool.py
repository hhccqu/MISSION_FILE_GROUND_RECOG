#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红色识别调试工具
用于分析图像中的红色分布，优化颜色识别参数
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

class ColorDebugTool:
    """颜色调试工具"""
    
    def __init__(self):
        """初始化调试工具"""
        pass
    
    def analyze_red_distribution(self, image_path: str, output_dir: str = "color_analysis"):
        """
        分析图像中红色的分布
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        filename = Path(image_path).stem
        
        print(f"分析图像: {filename}")
        
        # 1. 显示原始图像
        self._save_image(image, os.path.join(output_dir, f"{filename}_original.png"))
        
        # 2. 分析各个颜色空间
        self._analyze_bgr_space(image, output_dir, filename)
        self._analyze_hsv_space(image, output_dir, filename)
        self._analyze_lab_space(image, output_dir, filename)
        
        # 3. 测试不同的红色阈值
        self._test_red_thresholds(image, output_dir, filename)
        
        print(f"分析完成，结果保存在: {output_dir}")
    
    def _analyze_bgr_space(self, image: np.ndarray, output_dir: str, filename: str):
        """分析BGR颜色空间"""
        print("分析BGR颜色空间...")
        
        # 分离BGR通道
        b, g, r = cv2.split(image)
        
        # 保存各通道
        self._save_image(b, os.path.join(output_dir, f"{filename}_bgr_b.png"))
        self._save_image(g, os.path.join(output_dir, f"{filename}_bgr_g.png"))
        self._save_image(r, os.path.join(output_dir, f"{filename}_bgr_r.png"))
        
        # 计算红色强度 (R - G - B)
        red_intensity = np.clip(r.astype(np.int16) - g.astype(np.int16) - b.astype(np.int16), 0, 255).astype(np.uint8)
        self._save_image(red_intensity, os.path.join(output_dir, f"{filename}_bgr_red_intensity.png"))
        
        # BGR红色掩码测试
        bgr_masks = []
        
        # 测试多个BGR阈值
        bgr_thresholds = [
            ([0, 0, 100], [80, 80, 255]),      # 深红色
            ([0, 0, 150], [100, 100, 255]),    # 亮红色
            ([0, 0, 120], [60, 60, 255]),      # 纯红色
            ([0, 0, 80], [120, 120, 255]),     # 宽范围红色
        ]
        
        for i, (lower, upper) in enumerate(bgr_thresholds):
            mask = cv2.inRange(image, np.array(lower), np.array(upper))
            bgr_masks.append(mask)
            self._save_image(mask, os.path.join(output_dir, f"{filename}_bgr_mask_{i+1}.png"))
        
        # 组合BGR掩码
        combined_bgr = bgr_masks[0]
        for mask in bgr_masks[1:]:
            combined_bgr = cv2.bitwise_or(combined_bgr, mask)
        
        self._save_image(combined_bgr, os.path.join(output_dir, f"{filename}_bgr_combined.png"))
    
    def _analyze_hsv_space(self, image: np.ndarray, output_dir: str, filename: str):
        """分析HSV颜色空间"""
        print("分析HSV颜色空间...")
        
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 保存HSV通道
        self._save_image(h, os.path.join(output_dir, f"{filename}_hsv_h.png"))
        self._save_image(s, os.path.join(output_dir, f"{filename}_hsv_s.png"))
        self._save_image(v, os.path.join(output_dir, f"{filename}_hsv_v.png"))
        
        # HSV红色掩码测试
        hsv_masks = []
        
        # 测试多个HSV阈值
        hsv_thresholds = [
            # (H_min, S_min, V_min), (H_max, S_max, V_max)
            ([0, 50, 50], [10, 255, 255]),      # 红色范围1 (0-10度)
            ([170, 50, 50], [180, 255, 255]),   # 红色范围2 (170-180度)
            ([0, 100, 100], [10, 255, 255]),    # 高饱和度红色1
            ([170, 100, 100], [180, 255, 255]), # 高饱和度红色2
            ([0, 30, 30], [15, 255, 255]),      # 宽范围红色1
            ([165, 30, 30], [180, 255, 255]),   # 宽范围红色2
        ]
        
        for i, (lower, upper) in enumerate(hsv_thresholds):
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            hsv_masks.append(mask)
            self._save_image(mask, os.path.join(output_dir, f"{filename}_hsv_mask_{i+1}.png"))
        
        # 组合HSV掩码
        combined_hsv = hsv_masks[0]
        for mask in hsv_masks[1:]:
            combined_hsv = cv2.bitwise_or(combined_hsv, mask)
        
        self._save_image(combined_hsv, os.path.join(output_dir, f"{filename}_hsv_combined.png"))
    
    def _analyze_lab_space(self, image: np.ndarray, output_dir: str, filename: str):
        """分析LAB颜色空间"""
        print("分析LAB颜色空间...")
        
        # 转换到LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 保存LAB通道
        self._save_image(l, os.path.join(output_dir, f"{filename}_lab_l.png"))
        self._save_image(a, os.path.join(output_dir, f"{filename}_lab_a.png"))
        self._save_image(b, os.path.join(output_dir, f"{filename}_lab_b.png"))
        
        # LAB红色掩码测试
        lab_masks = []
        
        # 测试多个LAB阈值
        lab_thresholds = [
            ([20, 150, 150], [255, 255, 255]),  # 标准红色
            ([10, 140, 140], [255, 255, 255]),  # 宽范围红色
            ([30, 160, 160], [255, 255, 255]),  # 高A值红色
            ([0, 130, 130], [255, 255, 255]),   # 最宽范围红色
        ]
        
        for i, (lower, upper) in enumerate(lab_thresholds):
            mask = cv2.inRange(lab, np.array(lower), np.array(upper))
            lab_masks.append(mask)
            self._save_image(mask, os.path.join(output_dir, f"{filename}_lab_mask_{i+1}.png"))
        
        # 组合LAB掩码
        combined_lab = lab_masks[0]
        for mask in lab_masks[1:]:
            combined_lab = cv2.bitwise_or(combined_lab, mask)
        
        self._save_image(combined_lab, os.path.join(output_dir, f"{filename}_lab_combined.png"))
    
    def _test_red_thresholds(self, image: np.ndarray, output_dir: str, filename: str):
        """测试最终的红色识别效果"""
        print("测试综合红色识别效果...")
        
        # 使用改进的红色识别方法
        red_mask_bgr = self._create_optimized_red_mask_bgr(image)
        red_mask_hsv = self._create_optimized_red_mask_hsv(image)
        red_mask_lab = self._create_optimized_red_mask_lab(image)
        
        # 保存各个方法的结果
        self._save_image(red_mask_bgr, os.path.join(output_dir, f"{filename}_final_bgr.png"))
        self._save_image(red_mask_hsv, os.path.join(output_dir, f"{filename}_final_hsv.png"))
        self._save_image(red_mask_lab, os.path.join(output_dir, f"{filename}_final_lab.png"))
        
        # 组合最终结果
        final_mask = cv2.bitwise_or(red_mask_hsv, red_mask_bgr)
        final_mask = cv2.bitwise_or(final_mask, red_mask_lab)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        self._save_image(final_mask, os.path.join(output_dir, f"{filename}_final_result.png"))
        
        # 创建叠加图像显示识别效果
        overlay = image.copy()
        overlay[final_mask > 0] = [0, 0, 255]  # 将识别的红色区域标记为纯红色
        self._save_image(overlay, os.path.join(output_dir, f"{filename}_overlay.png"))
    
    def _create_optimized_red_mask_bgr(self, image: np.ndarray) -> np.ndarray:
        """优化的BGR红色掩码"""
        # 多层次的BGR红色检测
        masks = []
        
        # 深红色
        mask1 = cv2.inRange(image, np.array([0, 0, 120]), np.array([60, 60, 255]))
        masks.append(mask1)
        
        # 亮红色
        mask2 = cv2.inRange(image, np.array([0, 0, 150]), np.array([80, 80, 255]))
        masks.append(mask2)
        
        # 粉红色
        mask3 = cv2.inRange(image, np.array([100, 100, 150]), np.array([200, 200, 255]))
        masks.append(mask3)
        
        # 组合所有掩码
        combined = masks[0]
        for mask in masks[1:]:
            combined = cv2.bitwise_or(combined, mask)
        
        return combined
    
    def _create_optimized_red_mask_hsv(self, image: np.ndarray) -> np.ndarray:
        """优化的HSV红色掩码"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        masks = []
        
        # 红色范围1 (0-15度)
        mask1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([15, 255, 255]))
        masks.append(mask1)
        
        # 红色范围2 (165-180度)
        mask2 = cv2.inRange(hsv, np.array([165, 30, 30]), np.array([180, 255, 255]))
        masks.append(mask2)
        
        # 高饱和度红色
        mask3 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
        masks.append(mask3)
        
        mask4 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
        masks.append(mask4)
        
        # 组合所有掩码
        combined = masks[0]
        for mask in masks[1:]:
            combined = cv2.bitwise_or(combined, mask)
        
        return combined
    
    def _create_optimized_red_mask_lab(self, image: np.ndarray) -> np.ndarray:
        """优化的LAB红色掩码"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        masks = []
        
        # 标准红色
        mask1 = cv2.inRange(lab, np.array([10, 140, 140]), np.array([255, 255, 255]))
        masks.append(mask1)
        
        # 亮红色
        mask2 = cv2.inRange(lab, np.array([30, 150, 150]), np.array([255, 255, 255]))
        masks.append(mask2)
        
        # 组合掩码
        combined = masks[0]
        for mask in masks[1:]:
            combined = cv2.bitwise_or(combined, mask)
        
        return combined
    
    def _save_image(self, image: np.ndarray, path: str):
        """保存图像"""
        cv2.imwrite(path, image)
        print(f"保存图像: {path}")
    
    def batch_analyze(self, input_dir: str, output_dir: str = "color_analysis"):
        """批量分析目录中的所有图像"""
        input_path = Path(input_dir)
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]
        
        if not image_files:
            print(f"在目录 {input_dir} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        for image_file in image_files:
            print(f"\n处理: {image_file.name}")
            file_output_dir = os.path.join(output_dir, image_file.stem)
            self.analyze_red_distribution(str(image_file), file_output_dir)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='红色识别调试工具')
    parser.add_argument('--input', '-i', required=True, help='输入图像或目录')
    parser.add_argument('--output', '-o', default='color_analysis', help='输出目录')
    
    args = parser.parse_args()
    
    tool = ColorDebugTool()
    
    if os.path.isfile(args.input):
        # 单个文件
        tool.analyze_red_distribution(args.input, args.output)
    elif os.path.isdir(args.input):
        # 目录
        tool.batch_analyze(args.input, args.output)
    else:
        print(f"输入路径不存在: {args.input}")


if __name__ == "__main__":
    main() 