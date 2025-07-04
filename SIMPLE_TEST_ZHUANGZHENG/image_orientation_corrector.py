#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像方向自动转正系统
专门用于处理箭头/三角形标识图像的方向校正
基于尖端检测的几何方法实现
"""

import cv2
import numpy as np
import os
import logging
from typing import Tuple, Optional, List
import argparse
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageOrientationCorrector:
    """图像方向校正器"""
    
    def __init__(self, debug_mode: bool = False):
        """
        初始化校正器
        
        Args:
            debug_mode: 是否开启调试模式，保存中间处理结果
        """
        self.debug_mode = debug_mode
        self.debug_images = {}
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理：基于红色颜色识别进行二值化、形态学操作
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的二值图像
        """
        # 1. 确保图像是BGR格式
        if len(image.shape) != 3:
            logger.error("输入图像必须是彩色图像")
            return None
        
        # 保存原始图像用于调试
        if self.debug_mode:
            self.debug_images['original'] = image.copy()
        
        # 2. 高斯滤波去噪（在颜色分割前进行）
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        if self.debug_mode:
            self.debug_images['blurred'] = blurred
        
        # 3. 基于红色的颜色分割
        # 方法1: BGR颜色空间的红色范围
        red_mask_bgr = self._create_red_mask_bgr(blurred)
        
        # 方法2: HSV颜色空间的红色范围（更准确）
        red_mask_hsv = self._create_red_mask_hsv(blurred)
        
        # 方法3: LAB颜色空间的红色范围
        red_mask_lab = self._create_red_mask_lab(blurred)
        
        # 组合多个颜色空间的结果
        combined_mask = cv2.bitwise_or(red_mask_hsv, red_mask_bgr)
        combined_mask = cv2.bitwise_or(combined_mask, red_mask_lab)
        
        if self.debug_mode:
            self.debug_images['red_mask_bgr'] = red_mask_bgr
            self.debug_images['red_mask_hsv'] = red_mask_hsv
            self.debug_images['red_mask_lab'] = red_mask_lab
            self.debug_images['combined_mask'] = combined_mask
        
        # 4. 形态学操作优化掩码
        # 定义椭圆形结构元素
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 开运算：去除小噪点
        opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        
        # 闭运算：填充小孔洞
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
        
        # 膨胀操作：增强连通性
        dilated = cv2.dilate(closed, kernel_small, iterations=1)
        
        if self.debug_mode:
            self.debug_images['opened'] = opened
            self.debug_images['closed'] = closed
            self.debug_images['final_mask'] = dilated
        
        logger.info("基于红色颜色识别完成二值化处理")
        return dilated
    
    def _create_red_mask_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        在BGR颜色空间中创建红色掩码
        
        Args:
            image: BGR图像
            
        Returns:
            红色区域的二值掩码
        """
        # BGR颜色空间中红色的范围
        # 红色在BGR中：B分量低，G分量低，R分量高
        lower_red1 = np.array([0, 0, 100])      # 深红色下界
        upper_red1 = np.array([80, 80, 255])    # 深红色上界
        
        lower_red2 = np.array([0, 0, 150])      # 亮红色下界  Step 5: 修改YOLO检测器
        upper_red2 = np.array([100, 100, 255])  # 亮红色上界
        
        # 创建掩码
        mask1 = cv2.inRange(image, lower_red1, upper_red1)
        mask2 = cv2.inRange(image, lower_red2, upper_red2)
        
        # 合并掩码
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        return red_mask
    
    def _create_red_mask_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        在HSV颜色空间中创建红色掩码
        
        Args:
            image: BGR图像
            
        Returns:
            红色区域的二值掩码
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # HSV中红色的范围（红色在HSV中分布在0度和180度附近）
        # 第一个红色范围 (0-10度)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        
        # 第二个红色范围 (170-180度)
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # 创建掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # 合并掩码
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        return red_mask
    
    def _create_red_mask_lab(self, image: np.ndarray) -> np.ndarray:
        """
        在LAB颜色空间中创建红色掩码
        
        Args:
            image: BGR图像
            
        Returns:
            红色区域的二值掩码
        """
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # LAB颜色空间中红色的范围
        # L: 亮度, A: 绿-红轴(负值表示绿色，正值表示红色), B: 蓝-黄轴
        lower_red = np.array([20, 150, 150])    # L>=20, A>=150(偏红), B>=150(偏黄)
        upper_red = np.array([255, 255, 255])  # 上界
        
        # 创建掩码
        red_mask = cv2.inRange(lab, lower_red, upper_red)
        
        return red_mask
    
    def find_largest_contour(self, binary_image: np.ndarray) -> Optional[np.ndarray]:
        """
        找到最大的轮廓（假设为主要的箭头形状）
        
        Args:
            binary_image: 二值图像
            
        Returns:
            最大轮廓，如果没找到则返回None
        """
        # 查找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("未找到任何轮廓")
            return None
        
        # 找到面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 检查轮廓面积是否合理
        area = cv2.contourArea(largest_contour)
        if area < 100:
            logger.warning(f"最大轮廓面积过小: {area}")
            return None
        
        logger.info(f"找到最大轮廓，面积: {area}")
        
        # 轮廓近似简化
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if self.debug_mode:
            # 创建轮廓调试图像
            contour_img = np.zeros_like(binary_image)
            cv2.drawContours(contour_img, [largest_contour], -1, 255, 2)
            self.debug_images['contour'] = contour_img
        
        return largest_contour
    
    def find_tip_point(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        找到轮廓的尖端点
        
        Args:
            contour: 输入轮廓
            
        Returns:
            尖端点坐标 (x, y)，如果找不到则返回None
        """
        if contour is None or len(contour) < 3:
            return None
        
        # 方法1: 找到距离质心最远的点
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return None
        
        # 计算质心
        centroid_x = int(moments['m10'] / moments['m00'])
        centroid_y = int(moments['m01'] / moments['m00'])
        centroid = (centroid_x, centroid_y)
        
        # 找到距离质心最远的点
        max_distance = 0
        tip_point = None
        
        for point in contour:
            x, y = point[0]
            distance = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
            if distance > max_distance:
                max_distance = distance
                tip_point = (x, y)
        
        # 方法2: 使用凸包验证尖端点
        hull = cv2.convexHull(contour, returnPoints=True)
        
        # 在凸包点中找到距离质心最远的点作为尖端
        max_distance_hull = 0
        tip_point_hull = None
        
        for point in hull:
            x, y = point[0]
            distance = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
            if distance > max_distance_hull:
                max_distance_hull = distance
                tip_point_hull = (x, y)
        
        # 选择更可靠的尖端点
        final_tip = tip_point_hull if tip_point_hull else tip_point
        
        if self.debug_mode and final_tip:
            # 创建尖端点调试图像
            tip_img = np.zeros((contour.shape[0], contour.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(tip_img, [contour], -1, (0, 255, 0), 2)
            cv2.circle(tip_img, centroid, 5, (255, 0, 0), -1)  # 质心：蓝色
            cv2.circle(tip_img, final_tip, 5, (0, 0, 255), -1)  # 尖端：红色
            self.debug_images['tip_detection'] = tip_img
        
        logger.info(f"检测到尖端点: {final_tip}, 质心: {centroid}")
        return final_tip
    
    def calculate_rotation_angle(self, tip_point: Tuple[int, int], 
                               image_center: Tuple[int, int]) -> float:
        """
        计算使尖端朝上所需的旋转角度
        
        Args:
            tip_point: 尖端点坐标
            image_center: 图像中心坐标
            
        Returns:
            旋转角度（度）
        """
        # 计算从图像中心到尖端的向量
        dx = tip_point[0] - image_center[0]
        dy = tip_point[1] - image_center[1]
        
        # 计算与垂直向上方向的夹角
        # 注意：图像坐标系中y轴向下，所以使用-dy
        angle_rad = np.arctan2(dx, -dy)
        angle_deg = np.degrees(angle_rad)
        
        logger.info(f"计算旋转角度: {angle_deg:.2f}度")
        return angle_deg
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            
        Returns:
            旋转后的图像
        """
        # 获取图像中心
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 执行旋转
        rotated = cv2.warpAffine(image, rotation_matrix, 
                                (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        
        return rotated
    
    def correct_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        主要的方向校正函数
        
        Args:
            image: 输入图像
            
        Returns:
            校正后的图像和处理信息
        """
        info = {
            'success': False,
            'rotation_angle': 0,
            'tip_point': None,
            'contour_area': 0,
            'error_message': None
        }
        
        try:
            # 1. 预处理
            logger.info("开始图像预处理...")
            processed = self.preprocess_image(image)
            
            # 2. 找到最大轮廓
            logger.info("查找主要轮廓...")
            contour = self.find_largest_contour(processed)
            if contour is None:
                info['error_message'] = "未找到有效轮廓"
                return image, info
            
            info['contour_area'] = cv2.contourArea(contour)
            
            # 3. 找到尖端点
            logger.info("检测尖端点...")
            tip_point = self.find_tip_point(contour)
            if tip_point is None:
                info['error_message'] = "未找到尖端点"
                return image, info
            
            info['tip_point'] = tip_point
            
            # 4. 计算旋转角度
            image_center = (image.shape[1] // 2, image.shape[0] // 2)
            angle = self.calculate_rotation_angle(tip_point, image_center)
            info['rotation_angle'] = angle
            
            # 5. 旋转图像
            logger.info(f"旋转图像 {angle:.2f}度...")
            corrected_image = self.rotate_image(image, angle)
            
            info['success'] = True
            logger.info("图像方向校正完成")
            
            return corrected_image, info
            
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            logger.error(error_msg)
            info['error_message'] = error_msg
            return image, info
    
    def save_debug_images(self, output_dir: str, filename_prefix: str):
        """保存调试图像"""
        if not self.debug_mode or not self.debug_images:
            return
        
        debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        for stage, img in self.debug_images.items():
            debug_filename = f"{filename_prefix}_{stage}.png"
            debug_path = os.path.join(debug_dir, debug_filename)
            cv2.imwrite(debug_path, img)
            logger.info(f"保存调试图像: {debug_path}")


def process_images_batch(input_dir: str, output_dir: str, debug_mode: bool = False):
    """
    批量处理图像
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        debug_mode: 是否开启调试模式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图像格式
    supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图像文件
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                  if f.suffix.lower() in supported_formats]
    
    if not image_files:
        logger.warning(f"在目录 {input_dir} 中未找到支持的图像文件")
        return
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 创建处理器
    corrector = ImageOrientationCorrector(debug_mode=debug_mode)
    
    # 处理结果统计
    results = {
        'total': len(image_files),
        'success': 0,
        'failed': 0,
        'details': []
    }
    
    # 批量处理
    for image_file in image_files:
        logger.info(f"处理图像: {image_file.name}")
        
        try:
            # 读取图像
            image = cv2.imread(str(image_file))
            if image is None:
                logger.error(f"无法读取图像: {image_file}")
                results['failed'] += 1
                continue
            
            # 校正方向
            corrected_image, info = corrector.correct_orientation(image)
            
            # 保存结果
            output_filename = f"corrected_{image_file.name}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, corrected_image)
            
            # 保存调试图像
            if debug_mode:
                corrector.save_debug_images(output_dir, image_file.stem)
            
            # 记录结果
            result_info = {
                'filename': image_file.name,
                'success': info['success'],
                'rotation_angle': info['rotation_angle'],
                'tip_point': info['tip_point'],
                'contour_area': info['contour_area'],
                'error_message': info['error_message']
            }
            results['details'].append(result_info)
            
            if info['success']:
                results['success'] += 1
                logger.info(f"成功处理: {image_file.name}, 旋转角度: {info['rotation_angle']:.2f}度")
            else:
                results['failed'] += 1
                logger.warning(f"处理失败: {image_file.name}, 原因: {info['error_message']}")
                
        except Exception as e:
            logger.error(f"处理 {image_file.name} 时发生异常: {str(e)}")
            results['failed'] += 1
    
    # 输出处理结果统计
    logger.info("=" * 50)
    logger.info("批量处理完成")
    logger.info(f"总计: {results['total']} 个文件")
    logger.info(f"成功: {results['success']} 个文件")
    logger.info(f"失败: {results['failed']} 个文件")
    logger.info("=" * 50)
    
    # 保存处理报告
    save_processing_report(results, output_dir)
    
    return results


def save_processing_report(results: dict, output_dir: str):
    """保存处理报告"""
    report_path = os.path.join(output_dir, 'processing_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("图像方向校正处理报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"处理时间: {np.datetime64('now')}\n")
        f.write(f"总文件数: {results['total']}\n")
        f.write(f"成功处理: {results['success']}\n")
        f.write(f"处理失败: {results['failed']}\n\n")
        
        f.write("详细信息:\n")
        f.write("-" * 30 + "\n")
        
        for detail in results['details']:
            f.write(f"文件: {detail['filename']}\n")
            f.write(f"状态: {'成功' if detail['success'] else '失败'}\n")
            if detail['success']:
                f.write(f"旋转角度: {detail['rotation_angle']:.2f}度\n")
                f.write(f"尖端点: {detail['tip_point']}\n")
                f.write(f"轮廓面积: {detail['contour_area']}\n")
            else:
                f.write(f"错误信息: {detail['error_message']}\n")
            f.write("-" * 30 + "\n")
    
    logger.info(f"处理报告已保存至: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像方向自动校正工具')
    parser.add_argument('--input', '-i', required=True, help='输入图像目录')
    parser.add_argument('--output', '-o', required=True, help='输出目录')
    parser.add_argument('--debug', '-d', action='store_true', help='开启调试模式')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input):
        logger.error(f"输入目录不存在: {args.input}")
        return
    
    # 执行批量处理
    process_images_batch(args.input, args.output, args.debug)


if __name__ == "__main__":
    main() 