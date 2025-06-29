#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
箭头方向检测和修正算法
解决图像转正后箭头倒立导致OCR识别错误的问题
"""

import cv2
import numpy as np
import math
import easyocr
from typing import Tuple, Optional

class ArrowOrientationFixer:
    """箭头方向检测和修正器"""
    
    def __init__(self):
        """初始化"""
        self.ocr_reader = easyocr.Reader(['en', 'ch_sim'])
    
    def detect_arrow_orientation(self, image: np.ndarray) -> str:
        """
        检测箭头指向方向
        
        Args:
            image: 输入图像
            
        Returns:
            str: 箭头方向 ('up', 'down', 'left', 'right', 'unknown')
        """
        try:
            # 转换为HSV色彩空间检测红色箭头
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 红色检测
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([165, 50, 50])
            upper_red2 = np.array([179, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大轮廓（箭头）
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 50:
                    return self._analyze_arrow_direction(largest_contour, image.shape)
            
            return 'unknown'
            
        except Exception as e:
            print(f"⚠️ 箭头方向检测失败: {e}")
            return 'unknown'
    
    def _analyze_arrow_direction(self, contour: np.ndarray, image_shape: tuple) -> str:
        """
        分析箭头方向
        
        Args:
            contour: 箭头轮廓
            image_shape: 图像形状
            
        Returns:
            str: 箭头方向
        """
        try:
            # 计算轮廓的质心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return 'unknown'
            
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 分析轮廓形状特征
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            
            # 计算凸包缺陷
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if len(hull_indices) > 3:
                defects = cv2.convexityDefects(contour, hull_indices)
                
                if defects is not None:
                    # 找到最深的凸包缺陷点（箭头的凹陷部分）
                    max_defect_depth = 0
                    deepest_point = None
                    
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        far = tuple(contour[f][0])
                        depth = d / 256.0
                        
                        if depth > max_defect_depth:
                            max_defect_depth = depth
                            deepest_point = far
                    
                    if deepest_point:
                        # 根据凹陷点和质心的相对位置判断箭头方向
                        dx = deepest_point[0] - centroid_x
                        dy = deepest_point[1] - centroid_y
                        
                        # 箭头尖端应该与凹陷点相对
                        if abs(dx) > abs(dy):
                            return 'right' if dx < 0 else 'left'
                        else:
                            return 'down' if dy < 0 else 'up'
            
            # 备用方法：基于轮廓的几何特征
            return self._analyze_by_geometry(contour, centroid_x, centroid_y, image_shape)
            
        except Exception as e:
            print(f"⚠️ 箭头方向分析失败: {e}")
            return 'unknown'
    
    def _analyze_by_geometry(self, contour: np.ndarray, cx: int, cy: int, image_shape: tuple) -> str:
        """
        基于几何特征分析箭头方向
        
        Args:
            contour: 轮廓
            cx, cy: 质心坐标
            image_shape: 图像形状
            
        Returns:
            str: 箭头方向
        """
        try:
            h, w = image_shape[:2]
            
            # 将轮廓分为四个象限
            top_points = []
            bottom_points = []
            left_points = []
            right_points = []
            
            for point in contour:
                x, y = point[0]
                if y < cy:
                    top_points.append(point)
                else:
                    bottom_points.append(point)
                
                if x < cx:
                    left_points.append(point)
                else:
                    right_points.append(point)
            
            # 计算各方向的点密度和分布
            top_density = len(top_points) / max(1, len(contour))
            bottom_density = len(bottom_points) / max(1, len(contour))
            left_density = len(left_points) / max(1, len(contour))
            right_density = len(right_points) / max(1, len(contour))
            
            # 箭头尖端方向的点密度通常较小
            densities = {
                'up': top_density,
                'down': bottom_density,
                'left': left_density,
                'right': right_density
            }
            
            # 找到密度最小的方向（可能是箭头尖端）
            min_direction = min(densities, key=densities.get)
            
            return min_direction
            
        except Exception as e:
            print(f"⚠️ 几何分析失败: {e}")
            return 'unknown'
    
    def correct_arrow_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        修正箭头方向，确保箭头指向正确方向
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (修正后图像, 是否进行了修正)
        """
        try:
            # 检测当前箭头方向
            current_direction = self.detect_arrow_orientation(image)
            print(f"🧭 检测到箭头方向: {current_direction}")
            
            if current_direction == 'unknown':
                return image, False
            
            # 定义标准方向（假设正确的箭头应该指向右侧）
            target_direction = 'right'
            
            # 计算需要的旋转角度
            rotation_needed = self._calculate_rotation_angle(current_direction, target_direction)
            
            if rotation_needed == 0:
                return image, False
            
            # 执行旋转
            corrected_image = self._rotate_image(image, rotation_needed)
            
            # 验证旋转后的方向
            new_direction = self.detect_arrow_orientation(corrected_image)
            print(f"🔄 旋转{rotation_needed}度后箭头方向: {new_direction}")
            
            return corrected_image, True
            
        except Exception as e:
            print(f"⚠️ 箭头方向修正失败: {e}")
            return image, False
    
    def _calculate_rotation_angle(self, current: str, target: str) -> int:
        """
        计算从当前方向到目标方向需要的旋转角度
        
        Args:
            current: 当前方向
            target: 目标方向
            
        Returns:
            int: 旋转角度
        """
        direction_angles = {
            'right': 0,
            'down': 90,
            'left': 180,
            'up': 270
        }
        
        if current not in direction_angles or target not in direction_angles:
            return 0
        
        current_angle = direction_angles[current]
        target_angle = direction_angles[target]
        
        # 计算最短旋转路径
        rotation = (target_angle - current_angle) % 360
        if rotation > 180:
            rotation -= 360
        
        return rotation
    
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度
            
        Returns:
            np.ndarray: 旋转后图像
        """
        try:
            if angle == 0:
                return image
            
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # 计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 计算新的边界框大小
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # 调整旋转中心
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            # 执行旋转
            rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
            
            return rotated
            
        except Exception as e:
            print(f"⚠️ 图像旋转失败: {e}")
            return image
    
    def smart_rotate_with_ocr_validation(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        智能旋转，结合OCR验证选择最佳方向
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (最佳图像, 识别文本, 置信度)
        """
        try:
            # 测试四个主要方向
            test_angles = [0, 90, 180, 270]
            best_image = image
            best_text = ""
            best_confidence = 0.0
            best_angle = 0
            
            for angle in test_angles:
                # 旋转图像
                if angle == 0:
                    test_image = image
                else:
                    test_image = self._rotate_image(image, angle)
                
                # OCR识别
                try:
                    ocr_results = self.ocr_reader.readtext(test_image)
                    
                    if ocr_results:
                        # 找到最佳结果
                        best_result = max(ocr_results, key=lambda x: x[2])
                        text = best_result[1]
                        confidence = best_result[2]
                        
                        # 检查是否包含数字且不是倒立的数字
                        import re
                        if re.search(r'\d', text) and not self._is_upside_down_number(text):
                            if confidence > best_confidence:
                                best_image = test_image
                                best_text = text
                                best_confidence = confidence
                                best_angle = angle
                                
                except Exception as e:
                    continue
            
            print(f"🎯 最佳旋转角度: {best_angle}度, 识别文本: {best_text}, 置信度: {best_confidence:.2f}")
            return best_image, best_text, best_confidence
            
        except Exception as e:
            print(f"⚠️ 智能旋转失败: {e}")
            return image, "", 0.0
    
    def _is_upside_down_number(self, text: str) -> bool:
        """
        检测数字是否倒立
        
        Args:
            text: 识别的文本
            
        Returns:
            bool: 是否倒立
        """
        # 倒立的数字可能被识别为其他字符
        upside_down_patterns = [
            '6' in text and '9' not in text,  # 6倒立可能仍是6
            '9' in text and '6' not in text,  # 9倒立可能仍是9
            any(char in text.lower() for char in ['u', 'n', 'ɹ', 'ɐ']),  # 倒立字符
        ]
        
        return any(upside_down_patterns)

def test_arrow_orientation_fix():
    """测试箭头方向修正功能"""
    import os
    
    fixer = ArrowOrientationFixer()
    
    # 测试保存的图像
    original_dir = "rotation_test_results/original"
    
    if os.path.exists(original_dir):
        test_files = os.listdir(original_dir)[:5]  # 测试前5个文件
        
        for filename in test_files:
            if filename.endswith('.jpg'):
                filepath = os.path.join(original_dir, filename)
                image = cv2.imread(filepath)
                
                if image is not None:
                    print(f"\n🧪 测试图像: {filename}")
                    
                    # 方法1: 箭头方向检测和修正
                    corrected_image, was_corrected = fixer.correct_arrow_orientation(image)
                    if was_corrected:
                        output_path = f"corrected_arrow_{filename}"
                        cv2.imwrite(output_path, corrected_image)
                        print(f"   💾 箭头修正结果: {output_path}")
                    
                    # 方法2: 智能旋转与OCR验证
                    smart_image, text, confidence = fixer.smart_rotate_with_ocr_validation(image)
                    output_path = f"smart_rotated_{filename}"
                    cv2.imwrite(output_path, smart_image)
                    print(f"   💾 智能旋转结果: {output_path}")
                    print(f"   📝 识别结果: {text} (置信度: {confidence:.2f})")

if __name__ == "__main__":
    test_arrow_orientation_fix() 