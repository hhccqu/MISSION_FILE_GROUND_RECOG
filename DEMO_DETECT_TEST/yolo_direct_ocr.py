#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + 直接多方向OCR识别
无需箭头方向矫正，直接识别目标内的二位数
"""

import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from multi_direction_ocr import MultiDirectionOCR
import json

class YOLODirectOCR:
    """YOLO检测 + 直接OCR识别器"""
    
    def __init__(self):
        """初始化"""
        print("🚀 初始化YOLO直接OCR识别器...")
        
        # 加载YOLO模型
        model_path = "../weights/best1.pt"
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            # 尝试其他路径
            alt_paths = ["../weights/best.pt", "../weights/yolov8n.pt", "yolov8n.pt"]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"🔄 使用替代模型: {model_path}")
                    break
        
        try:
            self.yolo_model = YOLO(model_path)
            print(f"✅ YOLO模型加载成功: {model_path}")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            return
        
        # 初始化多方向OCR识别器
        self.ocr_processor = MultiDirectionOCR()
        print("✅ YOLO直接OCR识别器初始化完成")
    
    def process_images(self, test_dir="test_image_manuel", results_dir="yolo_direct_ocr_results"):
        """处理测试图像"""
        print("\n🧪 开始YOLO + 直接OCR测试...")
        
        if not os.path.exists(test_dir):
            print(f"❌ 测试目录不存在: {test_dir}")
            return
        
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 获取测试图像
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.png')]
        image_files.sort()
        
        print(f"📁 找到 {len(image_files)} 个测试图像")
        
        # 测试统计
        total_images = 0
        total_detections = 0
        successful_recognitions = 0
        high_confidence_count = 0
        
        # 详细结果
        detailed_results = []
        
        print("\n" + "="*100)
        
        for i, filename in enumerate(image_files, 1):
            print(f"\n🖼️  处理图像 {i}/{len(image_files)}: {filename}")
            print("-" * 80)
            
            filepath = os.path.join(test_dir, filename)
            image = cv2.imread(filepath)
            
            if image is None:
                print(f"❌ 无法加载图像: {filename}")
                continue
            
            total_images += 1
            base_name = os.path.splitext(filename)[0]
            
            print(f"📏 原始图像尺寸: {image.shape[1]}x{image.shape[0]}")
            
            # YOLO目标检测
            detections = self.detect_targets(image)
            
            if not detections:
                print("❌ 未检测到任何目标")
                continue
            
            print(f"🎯 检测到 {len(detections)} 个目标")
            
            # 处理每个检测目标
            image_result = {
                'filename': filename,
                'image_shape': image.shape,
                'detection_count': len(detections),
                'targets': []
            }
            
            for j, detection in enumerate(detections, 1):
                print(f"\n   📦 处理目标 {j}/{len(detections)}")
                
                # 提取目标区域
                target_roi = self.extract_target_roi(image, detection)
                
                if target_roi is None:
                    print(f"   ❌ 无法提取目标区域")
                    continue
                
                total_detections += 1
                
                print(f"   📏 目标尺寸: {target_roi.shape[1]}x{target_roi.shape[0]}")
                print(f"   📍 置信度: {detection['confidence']:.3f}")
                
                # 保存原始ROI
                roi_filename = f"{base_name}_target_{j}_original.jpg"
                roi_path = os.path.join(results_dir, roi_filename)
                cv2.imwrite(roi_path, target_roi)
                
                # 直接多方向OCR识别
                start_time = time.time()
                detected_number, ocr_confidence, ocr_details = self.ocr_processor.recognize_two_digit_number(target_roi)
                processing_time = time.time() - start_time
                
                # 统计成功识别
                if detected_number != "未识别":
                    successful_recognitions += 1
                    if ocr_confidence > 0.5:
                        high_confidence_count += 1
                
                # 保存识别结果的最佳方向图像
                if ocr_details['success']:
                    best_angle = ocr_details['best_result']['angle']
                    if best_angle != 0:
                        # 生成并保存最佳方向的图像
                        if best_angle == 90:
                            best_image = cv2.rotate(target_roi, cv2.ROTATE_90_CLOCKWISE)
                        elif best_angle == 180:
                            best_image = cv2.rotate(target_roi, cv2.ROTATE_180)
                        elif best_angle == 270:
                            best_image = cv2.rotate(target_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else:
                            best_image = target_roi
                        
                        best_filename = f"{base_name}_target_{j}_best_{best_angle}deg.jpg"
                        best_path = os.path.join(results_dir, best_filename)
                        cv2.imwrite(best_path, best_image)
                
                # 记录目标结果
                target_result = {
                    'target_id': j,
                    'bbox': detection['bbox'],
                    'yolo_confidence': detection['confidence'],
                    'detected_number': detected_number,
                    'ocr_confidence': ocr_confidence,
                    'best_angle': ocr_details['best_result']['angle'] if ocr_details['success'] else 0,
                    'processing_time': processing_time,
                    'all_directions': {}
                }
                
                # 记录所有方向的识别结果
                for angle, result in ocr_details['all_results'].items():
                    if 'error' not in result:
                        target_result['all_directions'][angle] = {
                            'text': result['text'],
                            'confidence': result['confidence'],
                            'two_digit_numbers': result['two_digit_numbers']
                        }
                
                image_result['targets'].append(target_result)
                
                # 显示结果
                self.display_target_result(target_result, j)
            
            detailed_results.append(image_result)
            
            # 创建图像摘要
            self.create_image_summary(image, detections, image_result['targets'], base_name, results_dir)
            
            print("=" * 80)
        
        # 保存详细结果
        self.save_detailed_results(detailed_results, results_dir)
        
        # 输出测试摘要
        self.print_test_summary(total_images, total_detections, successful_recognitions, high_confidence_count, results_dir)
    
    def detect_targets(self, image):
        """使用YOLO检测目标"""
        try:
            results = self.yolo_model(image, conf=0.25)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'area': (x2 - x1) * (y2 - y1)
                        })
            
            # 按置信度排序
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections
            
        except Exception as e:
            print(f"   ❌ YOLO检测失败: {e}")
            return []
    
    def extract_target_roi(self, image, detection):
        """提取目标区域"""
        try:
            x1, y1, x2, y2 = detection['bbox']
            
            # 计算智能边距
            target_width = x2 - x1
            target_height = y2 - y1
            width_margin = max(20, int(target_width * 0.3))
            height_margin = max(20, int(target_height * 0.3))
            
            if target_width < 50 or target_height < 50:
                width_margin = max(width_margin, 30)
                height_margin = max(height_margin, 30)
            
            h, w = image.shape[:2]
            
            x1_expanded = max(0, x1 - width_margin)
            y1_expanded = max(0, y1 - height_margin)
            x2_expanded = min(w, x2 + width_margin)
            y2_expanded = min(h, y2 + height_margin)
            
            roi = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
            
            if roi.size == 0:
                return None
            
            return roi
            
        except Exception as e:
            print(f"   ❌ 提取ROI失败: {e}")
            return None
    
    def display_target_result(self, result, target_idx):
        """显示目标处理结果"""
        print(f"   🎯 识别结果: {result['detected_number']}")
        print(f"   📊 OCR置信度: {result['ocr_confidence']:.3f}")
        print(f"   🔄 最佳角度: {result['best_angle']}°")
        print(f"   ⏱️  处理时间: {result['processing_time']:.2f}秒")
        
        # 显示所有方向的结果
        print(f"   📋 各方向识别:")
        for angle, dir_result in result['all_directions'].items():
            print(f"      {angle:3d}°: '{dir_result['text']}' (置信度: {dir_result['confidence']:.2f}) 二位数: {dir_result['two_digit_numbers']}")
    
    def create_image_summary(self, original_image, detections, target_results, base_name, results_dir):
        """创建图像处理摘要可视化"""
        try:
            annotated_image = original_image.copy()
            h, w = original_image.shape[:2]
            
            for i, (detection, target_result) in enumerate(zip(detections, target_results), 1):
                x1, y1, x2, y2 = detection['bbox']
                
                # 计算扩展框
                target_width = x2 - x1
                target_height = y2 - y1
                width_margin = max(20, int(target_width * 0.3))
                height_margin = max(20, int(target_height * 0.3))
                
                if target_width < 50 or target_height < 50:
                    width_margin = max(width_margin, 30)
                    height_margin = max(height_margin, 30)
                
                x1_expanded = max(0, x1 - width_margin)
                y1_expanded = max(0, y1 - height_margin)
                x2_expanded = min(w, x2 + width_margin)
                y2_expanded = min(h, y2 + height_margin)
                
                # 绘制检测框
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
                # 绘制扩展框
                color = (0, 255, 0) if target_result['detected_number'] != "未识别" else (0, 165, 255)
                cv2.rectangle(annotated_image, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), color, 2)
                
                # 添加标签
                label = f"T{i}: {target_result['detected_number']}"
                if target_result['best_angle'] != 0:
                    label += f" ({target_result['best_angle']}°)"
                
                info_label = f"Conf:{target_result['ocr_confidence']:.2f}"
                
                cv2.putText(annotated_image, label, (x1_expanded, y1_expanded-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(annotated_image, info_label, (x1_expanded, y1_expanded-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 保存标注图像
            summary_filename = f"{base_name}_summary.jpg"
            summary_path = os.path.join(results_dir, summary_filename)
            cv2.imwrite(summary_path, annotated_image)
            
            print(f"   💾 处理摘要已保存: {summary_filename}")
            
        except Exception as e:
            print(f"   ⚠️  创建摘要失败: {e}")
    
    def save_detailed_results(self, results, results_dir):
        """保存详细结果到JSON文件"""
        json_path = os.path.join(results_dir, "detailed_results.json")
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"📄 详细结果已保存: {json_path}")
            
        except Exception as e:
            print(f"⚠️ 保存详细结果失败: {e}")
    
    def print_test_summary(self, total_images, total_detections, successful_recognitions, high_confidence_count, results_dir):
        """输出测试摘要"""
        print(f"\n📋 YOLO + 直接OCR测试摘要")
        print("=" * 100)
        print(f"🖼️  处理图像总数: {total_images}")
        print(f"🎯 检测目标总数: {total_detections}")
        print(f"✅ 成功识别数量: {successful_recognitions} ({successful_recognitions/max(1,total_detections)*100:.1f}%)")
        print(f"📈 高置信度识别: {high_confidence_count} ({high_confidence_count/max(1,total_detections)*100:.1f}%)")
        print(f"📁 结果保存目录: {results_dir}/")
        
        # 创建文本报告
        self.create_text_report(total_images, total_detections, successful_recognitions, high_confidence_count, results_dir)
    
    def create_text_report(self, total_images, total_detections, successful_recognitions, high_confidence_count, results_dir):
        """创建文本测试报告"""
        report_path = os.path.join(results_dir, "test_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLO + 直接多方向OCR识别测试报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 测试统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"处理图像总数: {total_images}\n")
            f.write(f"检测目标总数: {total_detections}\n")
            f.write(f"成功识别数量: {successful_recognitions}\n")
            f.write(f"高置信度识别: {high_confidence_count}\n")
            f.write(f"识别成功率: {successful_recognitions/max(1,total_detections)*100:.1f}%\n")
            f.write(f"高置信度率: {high_confidence_count/max(1,total_detections)*100:.1f}%\n\n")
            
            f.write("🔧 技术方案\n")
            f.write("-" * 40 + "\n")
            f.write("1. YOLO目标检测: 使用best1.pt模型，置信度阈值0.25\n")
            f.write("2. 多方向OCR: 并行识别0°、90°、180°、270°四个方向\n")
            f.write("3. 智能选择: 基于置信度和二位数数量的综合评分\n")
            f.write("4. 无需矫正: 直接识别，避免箭头方向判断的复杂性\n\n")
            
            f.write("💡 算法优势\n")
            f.write("-" * 40 + "\n")
            f.write("- 简化流程: 跳过箭头方向检测步骤\n")
            f.write("- 并行处理: 四方向同时OCR，提高效率\n")
            f.write("- 鲁棒性强: 不依赖箭头颜色和形状特征\n")
            f.write("- 适应性好: 可处理任意方向的数字目标\n")
        
        print(f"📄 测试报告已保存: {report_path}")

def main():
    """主函数"""
    print("🚀 启动YOLO + 直接OCR识别测试")
    
    # 创建测试器
    processor = YOLODirectOCR()
    
    # 运行测试
    processor.process_images()
    
    print("\n✅ 测试完成!")

if __name__ == "__main__":
    main() 