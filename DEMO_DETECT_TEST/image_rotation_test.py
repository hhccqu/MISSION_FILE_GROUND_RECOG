#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像转正和OCR识别优化测试
从视频中截取YOLO检测到的目标图像，测试不同的转正算法和OCR识别效果
"""

import cv2
import numpy as np
import os
import time
import math
from ultralytics import YOLO
import easyocr
import re
from pathlib import Path

class ImageRotationTester:
    """图像转正测试器"""
    
    def __init__(self, video_path, output_dir="test_images"):
        """
        初始化测试器
        
        Args:
            video_path: 测试视频路径
            output_dir: 输出目录
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "original").mkdir(exist_ok=True)
        (self.output_dir / "rotated_red").mkdir(exist_ok=True)
        (self.output_dir / "rotated_edge").mkdir(exist_ok=True)
        (self.output_dir / "rotated_contour").mkdir(exist_ok=True)
        (self.output_dir / "comparison").mkdir(exist_ok=True)
        
        # 初始化模型
        self.model = None
        self.ocr_reader = None
        
        # 测试结果
        self.test_results = []
        
    def initialize_models(self):
        """初始化YOLO和OCR模型"""
        print("🎯 初始化YOLO模型...")
        model_paths = ['best1.pt', 'best.pt', 'yolov8n.pt', 'yolov8s.pt']
        
        for model_path in model_paths:
            try:
                self.model = YOLO(model_path)
                print(f"✅ YOLO模型加载成功: {model_path}")
                break
            except:
                continue
        
        if not self.model:
            print("❌ 无法加载YOLO模型")
            return False
        
        print("🔤 初始化OCR识别器...")
        try:
            self.ocr_reader = easyocr.Reader(['en', 'ch_sim'])
            print("✅ OCR识别器初始化成功")
            return True
        except Exception as e:
            print(f"❌ OCR初始化失败: {e}")
            return False
    
    def extract_targets_from_video(self, max_frames=100, frame_interval=10):
        """从视频中提取目标图像"""
        print(f"📹 开始从视频提取目标图像...")
        print(f"   视频路径: {self.video_path}")
        print(f"   最大帧数: {max_frames}")
        print(f"   帧间隔: {frame_interval}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {self.video_path}")
            return False
        
        frame_count = 0
        saved_count = 0
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 按间隔处理帧
                if frame_count % frame_interval != 0:
                    continue
                
                print(f"🔍 处理第{frame_count}帧...")
                
                # YOLO检测
                results = self.model(frame)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            if confidence > 0.25:  # 置信度阈值
                                # 提取目标ROI
                                roi = frame[int(y1):int(y2), int(x1):int(x2)]
                                
                                if roi.size > 0 and roi.shape[0] > 20 and roi.shape[1] > 20:
                                    # 保存原始图像
                                    filename = f"frame_{frame_count:04d}_target_{i}_conf_{confidence:.2f}"
                                    original_path = self.output_dir / "original" / f"{filename}.jpg"
                                    cv2.imwrite(str(original_path), roi)
                                    
                                    # 测试不同的转正方法
                                    self.test_rotation_methods(roi, filename)
                                    
                                    saved_count += 1
                                    print(f"   💾 保存目标图像: {filename}")
                
                if saved_count >= 50:  # 限制保存数量
                    break
        
        finally:
            cap.release()
        
        print(f"✅ 完成图像提取，共保存 {saved_count} 个目标图像")
        return True
    
    def test_rotation_methods(self, roi, filename):
        """测试不同的转正方法"""
        
        # 方法1: 基于红色箭头的转正（当前方法）
        rotated_red = self.rotate_by_red_arrow(roi.copy())
        red_path = self.output_dir / "rotated_red" / f"{filename}_red.jpg"
        cv2.imwrite(str(red_path), rotated_red)
        
        # 方法2: 基于边缘检测的转正
        rotated_edge = self.rotate_by_edge_detection(roi.copy())
        edge_path = self.output_dir / "rotated_edge" / f"{filename}_edge.jpg"
        cv2.imwrite(str(edge_path), rotated_edge)
        
        # 方法3: 基于轮廓分析的转正
        rotated_contour = self.rotate_by_contour_analysis(roi.copy())
        contour_path = self.output_dir / "rotated_contour" / f"{filename}_contour.jpg"
        cv2.imwrite(str(contour_path), rotated_contour)
        
        # 创建对比图像
        comparison = self.create_comparison_image(roi, rotated_red, rotated_edge, rotated_contour)
        comp_path = self.output_dir / "comparison" / f"{filename}_comparison.jpg"
        cv2.imwrite(str(comp_path), comparison)
        
        # OCR测试
        ocr_results = self.test_ocr_on_images(roi, rotated_red, rotated_edge, rotated_contour)
        
        # 记录测试结果
        self.test_results.append({
            'filename': filename,
            'ocr_results': ocr_results,
            'paths': {
                'original': str(self.output_dir / "original" / f"{filename}.jpg"),
                'red': str(red_path),
                'edge': str(edge_path),
                'contour': str(contour_path),
                'comparison': str(comp_path)
            }
        })
    
    def rotate_by_red_arrow(self, image):
        """方法1: 基于红色箭头的转正（当前使用的方法）"""
        if image is None or image.size == 0:
            return image
        
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 检测红色区域
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([30, 255, 255])
            lower_red2 = np.array([150, 30, 30])  
            upper_red2 = np.array([179, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # 形态学操作
            kernel = np.ones((3,3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                
                # 调整角度
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                
                if abs(angle) > 5:
                    center = (image.shape[1]//2, image.shape[0]//2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(image, rotation_matrix, 
                                           (image.shape[1], image.shape[0]))
                    return rotated
            
            return image
            
        except Exception as e:
            print(f"⚠️ 红色箭头转正失败: {e}")
            return image
    
    def rotate_by_edge_detection(self, image):
        """方法2: 基于边缘检测的转正"""
        if image is None or image.size == 0:
            return image
        
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 霍夫变换检测直线
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
            
            if lines is not None and len(lines) > 0:
                # 计算主要直线的角度
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = math.degrees(theta) - 90  # 转换为旋转角度
                    angles.append(angle)
                
                # 使用角度的中位数
                median_angle = np.median(angles)
                
                # 限制角度范围
                if median_angle < -45:
                    median_angle = 90 + median_angle
                elif median_angle > 45:
                    median_angle = median_angle - 90
                
                if abs(median_angle) > 5:
                    center = (image.shape[1]//2, image.shape[0]//2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(image, rotation_matrix, 
                                           (image.shape[1], image.shape[0]))
                    return rotated
            
            return image
            
        except Exception as e:
            print(f"⚠️ 边缘检测转正失败: {e}")
            return image
    
    def rotate_by_contour_analysis(self, image):
        """方法3: 基于轮廓分析的转正"""
        if image is None or image.size == 0:
            return image
        
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 计算轮廓的主要方向
                if len(largest_contour) >= 5:  # 需要至少5个点来拟合椭圆
                    ellipse = cv2.fitEllipse(largest_contour)
                    angle = ellipse[2]
                    
                    # 调整角度
                    if angle > 90:
                        angle = angle - 180
                    
                    if abs(angle) > 5:
                        center = (image.shape[1]//2, image.shape[0]//2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(image, rotation_matrix, 
                                               (image.shape[1], image.shape[0]))
                        return rotated
            
            return image
            
        except Exception as e:
            print(f"⚠️ 轮廓分析转正失败: {e}")
            return image
    
    def create_comparison_image(self, original, red_rotated, edge_rotated, contour_rotated):
        """创建对比图像"""
        try:
            # 调整所有图像到相同大小
            target_size = (150, 150)
            
            orig_resized = cv2.resize(original, target_size)
            red_resized = cv2.resize(red_rotated, target_size)
            edge_resized = cv2.resize(edge_rotated, target_size)
            contour_resized = cv2.resize(contour_rotated, target_size)
            
            # 创建2x2网格
            top_row = np.hstack([orig_resized, red_resized])
            bottom_row = np.hstack([edge_resized, contour_resized])
            comparison = np.vstack([top_row, bottom_row])
            
            # 添加标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (5, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, "Red Arrow", (155, 20), font, 0.5, (0, 0, 255), 1)
            cv2.putText(comparison, "Edge Detection", (5, 170), font, 0.5, (0, 255, 0), 1)
            cv2.putText(comparison, "Contour Analysis", (155, 170), font, 0.5, (255, 0, 0), 1)
            
            return comparison
            
        except Exception as e:
            print(f"⚠️ 创建对比图像失败: {e}")
            return original
    
    def test_ocr_on_images(self, original, red_rotated, edge_rotated, contour_rotated):
        """对所有图像进行OCR测试"""
        images = {
            'original': original,
            'red_rotated': red_rotated,
            'edge_rotated': edge_rotated,
            'contour_rotated': contour_rotated
        }
        
        results = {}
        
        for method, image in images.items():
            try:
                ocr_results = self.ocr_reader.readtext(image)
                
                # 提取最佳结果
                best_text = ""
                best_confidence = 0.0
                
                if ocr_results:
                    best_result = max(ocr_results, key=lambda x: x[2])
                    best_text = best_result[1]
                    best_confidence = best_result[2]
                
                # 提取数字
                numbers = re.findall(r'\d+', best_text)
                detected_number = numbers[0] if numbers else "未识别"
                
                results[method] = {
                    'text': best_text,
                    'confidence': best_confidence,
                    'number': detected_number,
                    'all_results': ocr_results
                }
                
            except Exception as e:
                results[method] = {
                    'text': "",
                    'confidence': 0.0,
                    'number': "错误",
                    'error': str(e)
                }
        
        return results
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n📊 生成测试报告...")
        
        report_path = self.output_dir / "test_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("图像转正和OCR识别测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 统计信息
            total_tests = len(self.test_results)
            f.write(f"总测试图像数: {total_tests}\n\n")
            
            # 各方法成功率统计
            methods = ['original', 'red_rotated', 'edge_rotated', 'contour_rotated']
            method_names = ['原始图像', '红色箭头转正', '边缘检测转正', '轮廓分析转正']
            
            for method, name in zip(methods, method_names):
                successful = sum(1 for result in self.test_results 
                               if result['ocr_results'][method]['number'] != "未识别" 
                               and result['ocr_results'][method]['number'] != "错误")
                success_rate = successful / total_tests * 100 if total_tests > 0 else 0
                f.write(f"{name}: {successful}/{total_tests} ({success_rate:.1f}%)\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
            
            # 详细结果
            f.write("详细测试结果:\n\n")
            
            for i, result in enumerate(self.test_results, 1):
                f.write(f"测试 {i}: {result['filename']}\n")
                f.write("-" * 30 + "\n")
                
                for method, name in zip(methods, method_names):
                    ocr_result = result['ocr_results'][method]
                    f.write(f"{name}:\n")
                    f.write(f"  识别文本: {ocr_result['text']}\n")
                    f.write(f"  提取数字: {ocr_result['number']}\n")
                    f.write(f"  置信度: {ocr_result['confidence']:.2f}\n")
                
                f.write("\n")
        
        print(f"✅ 测试报告已保存: {report_path}")
        
        # 打印简要统计
        print("\n📈 测试结果统计:")
        for method, name in zip(methods, method_names):
            successful = sum(1 for result in self.test_results 
                           if result['ocr_results'][method]['number'] != "未识别" 
                           and result['ocr_results'][method]['number'] != "错误")
            success_rate = successful / len(self.test_results) * 100 if self.test_results else 0
            print(f"   {name}: {successful}/{len(self.test_results)} ({success_rate:.1f}%)")
    
    def run_test(self, max_frames=100, frame_interval=10):
        """运行完整测试"""
        print("🚀 开始图像转正和OCR识别优化测试")
        print("=" * 60)
        
        # 初始化模型
        if not self.initialize_models():
            return False
        
        # 提取目标图像
        if not self.extract_targets_from_video(max_frames, frame_interval):
            return False
        
        # 生成测试报告
        self.generate_test_report()
        
        print(f"\n✅ 测试完成！结果保存在: {self.output_dir}")
        print(f"   原始图像: {self.output_dir}/original/")
        print(f"   红色箭头转正: {self.output_dir}/rotated_red/")
        print(f"   边缘检测转正: {self.output_dir}/rotated_edge/")
        print(f"   轮廓分析转正: {self.output_dir}/rotated_contour/")
        print(f"   对比图像: {self.output_dir}/comparison/")
        print(f"   测试报告: {self.output_dir}/test_report.txt")
        
        return True

def main():
    """主函数"""
    # 视频路径
    video_path = "../video2.mp4"  # 相对于DEMO_DETECT_TEST目录的路径
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        print("请确认视频路径是否正确")
        return
    
    # 创建测试器
    tester = ImageRotationTester(video_path, "rotation_test_results")
    
    # 运行测试
    tester.run_test(max_frames=200, frame_interval=5)

if __name__ == "__main__":
    main() 