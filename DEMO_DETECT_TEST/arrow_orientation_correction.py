#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统OpenCV方法进行箭头方向矫正
确保箭头尖端朝向正上方
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class ArrowOrientationCorrector:
    def __init__(self):
        """初始化箭头方向矫正器"""
        self.debug = True
        
    def detect_arrow_contour(self, image):
        """
        检测箭头轮廓
        
        Args:
            image: 输入图像
            
        Returns:
            largest_contour: 最大的轮廓（假设为箭头）
            mask: 箭头区域掩码
        """
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 基于调试结果，使用更宽泛的颜色范围
        # 包含红色、粉色和其他相似颜色
        lower_color = np.array([0, 30, 50])
        upper_color = np.array([180, 255, 255])
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask
            
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 过滤太小的轮廓
        if cv2.contourArea(largest_contour) < 100:
            return None, mask
        
        return largest_contour, mask
    
    def find_arrow_tip(self, contour):
        """
        检测箭头顶点
        
        Args:
            contour: 箭头轮廓
            
        Returns:
            tip_point: 箭头顶点坐标 (x, y)，如果检测失败返回None
        """
        
        # 首先尝试凸包缺陷方法
        tip_point = self.find_arrow_tip_by_convex_hull(contour)
        if tip_point is not None:
            return tip_point
        
        # 如果凸包方法失败，使用几何方法
        return self.find_arrow_tip_by_geometry(contour)
    
    def find_arrow_tip_by_convex_hull(self, contour):
        """
        使用凸包缺陷分析找到箭头顶点
        
        Args:
            contour: 箭头轮廓
            
        Returns:
            tip_point: 箭头顶点坐标 (x, y)，如果检测失败返回None
        """
        
        try:
            # 计算凸包
            hull = cv2.convexHull(contour, returnPoints=False)
            
            if len(hull) < 4:
                return None
            
            # 计算凸包缺陷
            defects = cv2.convexityDefects(contour, hull)
            
            if defects is None or len(defects) == 0:
                return None
            
            # 找到最深的缺陷点
            max_depth = 0
            deepest_defect = None
            
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0  # 转换为像素单位
                
                if depth > max_depth:
                    max_depth = depth
                    deepest_defect = f
            
            if deepest_defect is None:
                return None
            
            # 获取最深缺陷点的坐标
            defect_point = tuple(contour[deepest_defect][0])
            
            # 找到距离缺陷点最远的轮廓点作为箭头尖端
            contour_points = contour.reshape(-1, 2)
            distances = np.sqrt(np.sum((contour_points - defect_point)**2, axis=1))
            tip_idx = np.argmax(distances)
            
            return tuple(contour_points[tip_idx])
            
        except Exception as e:
            print(f"凸包方法失败: {e}")
            return None
    
    def find_arrow_tip_by_geometry(self, contour):
        """
        使用几何方法找到箭头顶点
        
        Args:
            contour: 箭头轮廓
            
        Returns:
            tip_point: 箭头顶点坐标 (x, y)，如果检测失败返回None
        """
        
        try:
            # 计算轮廓中心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None
            
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            # 找到距离中心最远的点
            contour_points = contour.reshape(-1, 2)
            distances = np.sqrt(np.sum((contour_points - [center_x, center_y])**2, axis=1))
            tip_idx = np.argmax(distances)
            
            return tuple(contour_points[tip_idx])
            
        except Exception as e:
            print(f"几何方法失败: {e}")
            return None
    
    def calculate_rotation_angle(self, tip_point, center_point, image_shape):
        """
        计算旋转角度，使tip与center的连线垂直且tip在上方
        
        Args:
            tip_point: 箭头顶点坐标 (x, y)
            center_point: 箭头中心坐标 (x, y)  
            image_shape: 图像形状 (height, width)
            
        Returns:
            rotation_angle: 需要旋转的角度（度）
        """
        
        # 计算从center到tip的向量
        dx = tip_point[0] - center_point[0]
        dy = tip_point[1] - center_point[1]  # 注意：图像坐标系中y向下为正
        
        print(f"🔍 Center到Tip向量: dx={dx}, dy={dy}")
        
        # 计算当前连线与垂直向上方向的夹角
        # 垂直向上是(0, -1)，使用atan2计算角度
        current_angle = np.degrees(np.arctan2(dx, -dy))  # 相对于垂直向上方向
        print(f"🔍 当前连线角度（相对垂直向上）: {current_angle:.1f}°")
        
        # 目标：连线应该垂直向上，即角度为0度
        target_angle = 0.0
        
        # 计算需要旋转的角度
        rotation_angle = target_angle - current_angle
        
        # 将角度标准化到[-180, 180]范围
        while rotation_angle > 180:
            rotation_angle -= 360
        while rotation_angle < -180:
            rotation_angle += 360
        
        print(f"🔍 目标角度: {target_angle}°")
        print(f"🔍 需要旋转: {rotation_angle:.1f}°")
        
        # 检查tip是否已经在center上方
        if dy < 0:  # tip在center上方（y坐标更小）
            print("✅ Tip已在Center上方")
        else:
            print("⚠️ Tip在Center下方，需要调整方向")
            # 如果tip在center下方，旋转角度需要调整180度
            if rotation_angle > 0:
                rotation_angle -= 180
            else:
                rotation_angle += 180
                
            # 重新标准化
            while rotation_angle > 180:
                rotation_angle -= 360
            while rotation_angle < -180:
                rotation_angle += 360
                
            print(f"🔍 调整后旋转角度: {rotation_angle:.1f}°")
        
        return rotation_angle
    
    def verify_tip_position(self, rotated_image, original_tip, rotation_matrix):
        """
        验证旋转后tip-center连线是否垂直且tip在上方
        
        Args:
            rotated_image: 旋转后的图像
            original_tip: 原始顶点坐标
            rotation_matrix: 旋转矩阵
            
        Returns:
            is_correct: 是否正确旋转
            verification_info: 验证信息
        """
        
        try:
            # 在旋转后的图像中重新检测箭头
            contours = self.detect_arrow_contours(rotated_image)
            if not contours:
                return False, "未检测到轮廓"
            
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算新的中心点
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return False, "无法计算中心点"
            
            new_center_x = int(M["m10"] / M["m00"])
            new_center_y = int(M["m01"] / M["m00"])
            new_center = (new_center_x, new_center_y)
            
            # 检测新的顶点
            new_tip = self.find_arrow_tip(largest_contour)
            if new_tip is None:
                return False, "无法检测顶点"
            
            # 计算tip-center连线的角度
            dx = new_tip[0] - new_center[0]
            dy = new_tip[1] - new_center[1]
            
            # 计算连线与垂直方向的夹角
            line_angle = np.degrees(np.arctan2(dx, -dy))  # 使用-dy因为我们要相对于向上的垂直方向
            
            # 检查是否接近垂直（允许一定误差）
            angle_tolerance = 15  # 度
            is_vertical = abs(line_angle) <= angle_tolerance
            
            # 检查tip是否在center上方
            is_tip_above = dy < 0
            
            verification_info = {
                'new_tip': new_tip,
                'new_center': new_center,
                'line_angle': line_angle,
                'is_vertical': is_vertical,
                'is_tip_above': is_tip_above
            }
            
            is_correct = is_vertical and is_tip_above
            
            print(f"🔍 验证结果:")
            print(f"   新顶点: {new_tip}")
            print(f"   新中心: {new_center}")
            print(f"   连线角度: {line_angle:.1f}° (相对垂直)")
            print(f"   是否垂直: {is_vertical} (误差≤{angle_tolerance}°)")
            print(f"   Tip在上方: {is_tip_above}")
            print(f"   整体正确: {is_correct}")
            
            return is_correct, verification_info
            
        except Exception as e:
            print(f"验证过程出错: {e}")
            return False, f"验证失败: {str(e)}"
    
    def rotate_image(self, image, angle, center=None):
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            center: 旋转中心，如果为None则使用图像中心
            
        Returns:
            rotated_image: 旋转后的图像
            rotation_matrix: 旋转矩阵
        """
        h, w = image.shape[:2]
        
        if center is None:
            center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新的边界框大小
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        
        # 调整旋转矩阵以适应新的图像大小
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # 执行旋转
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        
        return rotated, rotation_matrix
    
    def correct_arrow_orientation(self, image_path, output_dir=None):
        """
        矫正箭头方向，确保箭头顶点位于图像最高位置
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": "无法读取图像"}
            
            print(f"📸 处理图像: {image_path}")
            
            # 检测箭头轮廓
            contours = self.detect_arrow_contours(image)
            if not contours:
                return {"success": False, "error": "未检测到箭头轮廓"}
            
            # 找到最大的轮廓（假设为箭头）
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            print(f"📏 箭头轮廓面积: {contour_area:.0f} 像素")
            
            # 计算轮廓中心
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return {"success": False, "error": "无法计算轮廓中心"}
            
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            center_point = (center_x, center_y)
            print(f"📍 轮廓中心: ({center_x}, {center_y})")
            
            # 检测箭头顶点
            tip_point = self.find_arrow_tip(largest_contour)
            if tip_point is None:
                return {"success": False, "error": "无法检测箭头顶点"}
            
            print(f"🎯 箭头顶点: ({tip_point[0]}, {tip_point[1]})")
            
            # 检查tip-center连线是否已经垂直且tip在上方
            dx = tip_point[0] - center_point[0]
            dy = tip_point[1] - center_point[1]
            
            # 计算连线与垂直方向的夹角
            line_angle = np.degrees(np.arctan2(dx, -dy))  # 相对于向上垂直方向
            
            print(f"📊 Tip-Center连线角度: {line_angle:.1f}° (相对垂直)")
            print(f"📊 Tip是否在Center上方: {dy < 0}")
            
            # 检查是否已经正确（垂直且tip在上方）
            angle_tolerance = 10  # 角度误差容忍度
            is_already_vertical = abs(line_angle) <= angle_tolerance
            is_tip_above = dy < 0
            
            if is_already_vertical and is_tip_above:
                print("✅ 箭头连线已经垂直且tip在上方，无需旋转")
                
                # 保存原图作为结果
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    corrected_path = os.path.join(output_dir, f"{base_name}_corrected.jpg")
                    cv2.imwrite(corrected_path, image)
                    
                    return {
                        "success": True,
                        "original_image": image_path,
                        "corrected_image": corrected_path,
                        "tip_point": tip_point,
                        "center_point": center_point,
                        "rotation_angle": 0.0,
                        "tip_angle": 0.0,
                        "contour_area": contour_area,
                        "already_correct": True
                    }
            
            # 计算旋转角度
            rotation_angle = self.calculate_rotation_angle(tip_point, center_point, image.shape)
            print(f"🔄 需要旋转角度: {rotation_angle:.1f}°")
            
            # 执行旋转
            rotated_image, rotation_matrix = self.rotate_image(image, rotation_angle, center_point)
            
            # 验证旋转结果
            is_correct, verification_info = self.verify_tip_position(rotated_image, tip_point, rotation_matrix)
            
            if not is_correct:
                print("⚠️ 第一次旋转未达到预期，尝试调整...")
                # 尝试相反方向旋转
                alternative_angle = rotation_angle + 180
                if alternative_angle > 180:
                    alternative_angle -= 360
                
                rotated_image, rotation_matrix = self.rotate_image(image, alternative_angle, center_point)
                is_correct, verification_info = self.verify_tip_position(rotated_image, tip_point, rotation_matrix)
                
                if is_correct:
                    rotation_angle = alternative_angle
                    print(f"✅ 调整后旋转角度: {rotation_angle:.1f}°")
                else:
                    print("⚠️ 旋转调整后仍未完全准确，但继续使用当前结果")
            
            print(f"✅ 旋转完成，新顶点y坐标: {verification_info['new_tip'][1]}")
            
            # 保存结果
            result = {
                "success": True,
                "original_image": image_path,
                "tip_point": verification_info['new_tip'],
                "center_point": verification_info['new_center'],
                "rotation_angle": rotation_angle,
                "tip_angle": 0.0,  # 保持兼容性
                "contour_area": contour_area,
                "rotated_image": rotated_image,
                "already_correct": False
            }
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # 保存矫正后的图像
                corrected_path = os.path.join(output_dir, f"{base_name}_corrected.jpg")
                cv2.imwrite(corrected_path, rotated_image)
                result["corrected_image"] = corrected_path
                
                # 创建处理过程可视化
                process_image = self.create_process_visualization(
                    image, rotated_image, largest_contour, verification_info['new_tip'], verification_info['new_center'], rotation_angle
                )
                process_path = os.path.join(output_dir, f"{base_name}_correction_process.jpg")
                cv2.imwrite(process_path, process_image)
                result["process_image"] = process_path
                
                print(f"💾 矫正结果保存至: {corrected_path}")
                print(f"🖼️ 处理过程保存至: {process_path}")
            
            return result
            
        except Exception as e:
            print(f"❌ 处理出错: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def visualize_correction_process(self, original, mask, contour, tip_point, 
                                   corrected, rotation_angle, output_dir, base_name):
        """
        可视化矫正过程
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始图像
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("原始图像")
        axes[0, 0].axis('off')
        
        # 箭头掩码
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title("箭头检测掩码")
        axes[0, 1].axis('off')
        
        # 轮廓和尖端标记
        contour_img = original.copy()
        cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
        if tip_point is not None:
            cv2.circle(contour_img, tuple(map(int, tip_point)), 5, (255, 0, 0), -1)
        axes[1, 0].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"轮廓和尖端\n旋转角度: {rotation_angle:.1f}°")
        axes[1, 0].axis('off')
        
        # 矫正后图像
        axes[1, 1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("矫正后图像")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存可视化结果
        viz_path = os.path.join(output_dir, f"{base_name}_correction_process.jpg")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 可视化结果保存至: {viz_path}")

    def create_process_visualization(self, original_image, corrected_image, contour, tip_point, center_point, rotation_angle):
        """
        创建处理过程可视化图像
        
        Args:
            original_image: 原始图像
            corrected_image: 矫正后图像
            contour: 检测到的轮廓
            tip_point: 箭头顶点
            center_point: 轮廓中心点
            rotation_angle: 旋转角度
            
        Returns:
            process_image: 可视化图像
        """
        
        # 创建原始图像的副本用于标注
        original_annotated = original_image.copy()
        
        # 在原始图像上绘制轮廓
        cv2.drawContours(original_annotated, [contour], -1, (0, 255, 0), 2)
        
        # 标记顶点
        cv2.circle(original_annotated, tip_point, 8, (0, 0, 255), -1)
        cv2.putText(original_annotated, "Tip", (tip_point[0] + 10, tip_point[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 标记中心点
        cv2.circle(original_annotated, center_point, 6, (255, 0, 0), -1)
        cv2.putText(original_annotated, "Center", (center_point[0] + 10, center_point[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 绘制从中心到顶点的箭头
        cv2.arrowedLine(original_annotated, center_point, tip_point, (255, 255, 0), 3)
        
        # 添加垂直参考线（从center向上延伸）
        ref_line_length = 50
        ref_top = (center_point[0], max(0, center_point[1] - ref_line_length))
        cv2.line(original_annotated, center_point, ref_top, (0, 255, 0), 2)
        cv2.putText(original_annotated, "Vertical Ref", (ref_top[0] + 5, ref_top[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 计算并显示连线角度
        dx = tip_point[0] - center_point[0]
        dy = tip_point[1] - center_point[1]
        line_angle = np.degrees(np.arctan2(dx, -dy))
        angle_text = f"Angle: {line_angle:.1f}°"
        cv2.putText(original_annotated, angle_text, (center_point[0] - 50, center_point[1] + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 调整图像大小使其相同
        h1, w1 = original_annotated.shape[:2]
        h2, w2 = corrected_image.shape[:2]
        
        # 使用较大的尺寸
        max_h = max(h1, h2)
        max_w = max(w1, w2)
        
        # 创建白色背景
        original_resized = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
        corrected_resized = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
        
        # 将图像居中放置
        y_offset1 = (max_h - h1) // 2
        x_offset1 = (max_w - w1) // 2
        original_resized[y_offset1:y_offset1+h1, x_offset1:x_offset1+w1] = original_annotated
        
        y_offset2 = (max_h - h2) // 2
        x_offset2 = (max_w - w2) // 2
        corrected_resized[y_offset2:y_offset2+h2, x_offset2:x_offset2+w2] = corrected_image
        
        # 水平拼接两个图像
        combined = np.hstack([original_resized, corrected_resized])
        
        # 添加标题和信息
        title_height = 80
        info_height = 60
        total_height = combined.shape[0] + title_height + info_height
        
        # 创建最终图像
        final_image = np.ones((total_height, combined.shape[1], 3), dtype=np.uint8) * 255
        
        # 添加标题
        cv2.putText(final_image, "Arrow Orientation Correction", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # 放置拼接的图像
        final_image[title_height:title_height+combined.shape[0], :] = combined
        
        # 添加图像标签
        cv2.putText(final_image, "Original (with annotations)", (20, title_height + combined.shape[0] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(final_image, "Corrected (tip-center vertical)", (combined.shape[1]//2 + 20, title_height + combined.shape[0] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 添加旋转角度信息
        angle_text = f"Rotation angle: {rotation_angle:.1f} degrees"
        cv2.putText(final_image, angle_text, (20, title_height + combined.shape[0] + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return final_image

    def detect_arrow_contours(self, image):
        """
        检测图像中的箭头轮廓
        
        Args:
            image: 输入图像
            
        Returns:
            contours: 检测到的轮廓列表
        """
        
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义箭头颜色范围（基于之前的调试结果）
        lower_bound = np.array([0, 30, 50])
        upper_bound = np.array([179, 255, 255])
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 形态学操作清理掩码
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓（面积和形状）
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 最小面积阈值
                # 检查轮廓的紧凑性
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.1 < compactness < 0.8:  # 箭头形状的紧凑性范围
                        filtered_contours.append(contour)
        
        return filtered_contours

if __name__ == "__main__":
    # 简单测试
    corrector = ArrowOrientationCorrector()
    print("✅ ArrowOrientationCorrector 类创建成功") 