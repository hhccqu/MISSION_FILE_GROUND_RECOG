#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO目标检测 + 箭头方向修正综合测试
先使用best1模型检测图像中的目标，然后对每个目标进行箭头方向修正测试
"""

import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from arrow_orientation_fix import ArrowOrientationFixer
import easyocr

class YOLOArrowTester:
    """YOLO检测与箭头修正综合测试器"""
    
    def __init__(self):
        """初始化"""
        print("🚀 初始化YOLO箭头测试器...")
        
        # 加载YOLO模型
        model_path = "../weights/best1.pt"
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            # 尝试其他路径
            alt_paths = ["../weights/best.pt", "../weights/yolov8n.pt"]
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
        
        # 初始化箭头方向修正器
        self.arrow_fixer = ArrowOrientationFixer()
        print("✅ 箭头方向修正器初始化完成")
        
        # 初始化OCR
        self.ocr_reader = easyocr.Reader(['en', 'ch_sim'])
        print("✅ OCR识别器初始化完成")
    
    def test_manual_images_with_yolo(self):
        """使用YOLO检测测试手动图像"""
        print("\n🧪 开始YOLO + 箭头修正综合测试...")
        
        # 测试图像目录
        test_dir = "test_image_manuel"
        
        if not os.path.exists(test_dir):
            print(f"❌ 测试目录不存在: {test_dir}")
            return
        
        # 获取所有PNG图像
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.png')]
        image_files.sort()
        
        print(f"📁 找到 {len(image_files)} 个测试图像")
        
        # 创建结果目录
        results_dir = "yolo_arrow_test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 测试统计
        total_images = 0
        total_detections = 0
        successful_corrections = 0
        high_confidence_ocr = 0
        
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
            
            # 步骤1: YOLO目标检测
            detections = self.detect_targets(image)
            
            if not detections:
                print("❌ 未检测到任何目标")
                continue
            
            print(f"🎯 检测到 {len(detections)} 个目标")
            
            # 步骤2: 对每个检测目标进行处理
            image_results = []
            
            for j, detection in enumerate(detections, 1):
                print(f"\n   📦 处理目标 {j}/{len(detections)}")
                
                # 提取目标区域
                target_roi = self.extract_target_roi(image, detection)
                
                if target_roi is None:
                    print(f"   ❌ 无法提取目标区域")
                    continue
                
                total_detections += 1
                
                # 保存原始ROI
                roi_filename = f"{base_name}_target_{j}_original.jpg"
                roi_path = os.path.join(results_dir, roi_filename)
                cv2.imwrite(roi_path, target_roi)
                
                print(f"   📏 目标尺寸: {target_roi.shape[1]}x{target_roi.shape[0]}")
                print(f"   📍 置信度: {detection['confidence']:.3f}")
                
                # 步骤3: 箭头方向检测和修正
                result = self.process_target_arrow(target_roi, base_name, j, results_dir)
                
                if result['corrected']:
                    successful_corrections += 1
                
                if result['ocr_confidence'] > 0.5:
                    high_confidence_ocr += 1
                
                image_results.append(result)
                
                # 显示处理结果
                self.display_target_result(result, j)
            
            # 步骤4: 创建综合可视化
            self.create_image_summary(image, detections, image_results, base_name, results_dir)
            
            print("=" * 80)
        
        # 输出测试摘要
        self.print_test_summary(total_images, total_detections, successful_corrections, high_confidence_ocr, results_dir)
    
    def detect_targets(self, image):
        """使用YOLO检测目标"""
        try:
            # 运行YOLO检测
            results = self.yolo_model(image, conf=0.25)  # 置信度阈值0.25
            
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
            
            # 计算目标的宽度和高度
            target_width = x2 - x1
            target_height = y2 - y1
            
            # 智能边距计算：基于目标大小的百分比 + 固定最小边距
            width_margin = max(20, int(target_width * 0.3))  # 宽度30%或至少20像素
            height_margin = max(20, int(target_height * 0.3))  # 高度30%或至少20像素
            
            # 对于小目标，使用更大的边距
            if target_width < 50 or target_height < 50:
                width_margin = max(width_margin, 30)
                height_margin = max(height_margin, 30)
                print(f"   📏 小目标检测，使用扩大边距: {width_margin}x{height_margin}")
            
            h, w = image.shape[:2]
            
            # 应用边距并确保不超出图像边界
            x1_expanded = max(0, x1 - width_margin)
            y1_expanded = max(0, y1 - height_margin)
            x2_expanded = min(w, x2 + width_margin)
            y2_expanded = min(h, y2 + height_margin)
            
            roi = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
            
            # 检查ROI有效性
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                return None
            
            # 输出扩展信息
            original_size = f"{target_width}x{target_height}"
            expanded_size = f"{roi.shape[1]}x{roi.shape[0]}"
            print(f"   📦 目标框扩展: {original_size} → {expanded_size} (边距: {width_margin}x{height_margin})")
            
            return roi
            
        except Exception as e:
            print(f"   ❌ ROI提取失败: {e}")
            return None
    
    def process_target_arrow(self, target_roi, base_name, target_idx, results_dir):
        """处理目标的箭头方向"""
        result = {
            'target_idx': target_idx,
            'corrected': False,
            'arrow_direction': 'unknown',
            'ocr_text': '',
            'ocr_confidence': 0.0,
            'original_ocr_text': '',
            'original_ocr_confidence': 0.0,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # 原始图像OCR
            try:
                original_ocr = self.ocr_reader.readtext(target_roi)
                if original_ocr:
                    best_original = max(original_ocr, key=lambda x: x[2])
                    result['original_ocr_text'] = best_original[1]
                    result['original_ocr_confidence'] = best_original[2]
            except:
                pass
            
            # 箭头方向检测和修正
            corrected_image, was_corrected = self.arrow_fixer.correct_arrow_orientation(target_roi)
            result['corrected'] = was_corrected
            
            if was_corrected:
                # 保存修正后的图像
                corrected_filename = f"{base_name}_target_{target_idx}_corrected.jpg"
                corrected_path = os.path.join(results_dir, corrected_filename)
                cv2.imwrite(corrected_path, corrected_image)
            
            # 智能旋转与OCR验证
            smart_image, ocr_text, ocr_confidence = self.arrow_fixer.smart_rotate_with_ocr_validation(target_roi)
            result['ocr_text'] = ocr_text
            result['ocr_confidence'] = ocr_confidence
            
            # 保存智能处理结果
            smart_filename = f"{base_name}_target_{target_idx}_smart.jpg"
            smart_path = os.path.join(results_dir, smart_filename)
            cv2.imwrite(smart_path, smart_image)
            
            # 检测箭头方向
            result['arrow_direction'] = self.arrow_fixer.detect_arrow_orientation(target_roi)
            
        except Exception as e:
            print(f"   ❌ 箭头处理失败: {e}")
        
        result['processing_time'] = time.time() - start_time
        
        return result
    
    def display_target_result(self, result, target_idx):
        """显示目标处理结果"""
        print(f"   🧭 箭头方向: {result['arrow_direction']}")
        print(f"   🔄 是否修正: {'是' if result['corrected'] else '否'}")
        print(f"   📝 原始OCR: '{result['original_ocr_text']}' (置信度: {result['original_ocr_confidence']:.2f})")
        print(f"   🎯 智能OCR: '{result['ocr_text']}' (置信度: {result['ocr_confidence']:.2f})")
        print(f"   ⏱️  处理时间: {result['processing_time']:.2f}秒")
        
        # 分析改进效果
        if result['ocr_confidence'] > result['original_ocr_confidence']:
            improvement = result['ocr_confidence'] - result['original_ocr_confidence']
            print(f"   📈 OCR改进: +{improvement:.2f}")
        elif result['ocr_confidence'] < result['original_ocr_confidence']:
            decline = result['original_ocr_confidence'] - result['ocr_confidence']
            print(f"   📉 OCR下降: -{decline:.2f}")
        else:
            print(f"   📊 OCR无变化")
    
    def create_image_summary(self, original_image, detections, results, base_name, results_dir):
        """创建图像处理摘要可视化"""
        try:
            # 在原图上标注检测结果
            annotated_image = original_image.copy()
            h, w = original_image.shape[:2]
            
            for i, (detection, result) in enumerate(zip(detections, results), 1):
                x1, y1, x2, y2 = detection['bbox']
                
                # 计算扩展框（与extract_target_roi中的逻辑保持一致）
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
                
                # 绘制原始YOLO检测框（红色虚线）
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
                # 绘制扩展处理框（绿色实线）
                color = (0, 255, 0) if result['corrected'] else (255, 165, 0)  # 绿色或橙色
                cv2.rectangle(annotated_image, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), color, 2)
                
                # 添加标签
                label = f"T{i}: {result['arrow_direction']}"
                if result['ocr_text']:
                    label += f" [{result['ocr_text']}]"
                
                # 显示置信度和尺寸信息
                info_label = f"Conf:{detection['confidence']:.2f} Size:{target_width}x{target_height}"
                
                cv2.putText(annotated_image, label, (x1_expanded, y1_expanded-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(annotated_image, info_label, (x1_expanded, y1_expanded-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 添加图例
            legend_y = 30
            cv2.putText(annotated_image, "Legend:", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(annotated_image, (10, legend_y+10), (30, legend_y+20), (0, 0, 255), 1)
            cv2.putText(annotated_image, "YOLO Detection", (35, legend_y+18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(annotated_image, (10, legend_y+25), (30, legend_y+35), (0, 255, 0), 2)
            cv2.putText(annotated_image, "Expanded ROI", (35, legend_y+33), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 保存标注图像
            summary_filename = f"{base_name}_summary.jpg"
            summary_path = os.path.join(results_dir, summary_filename)
            cv2.imwrite(summary_path, annotated_image)
            
            print(f"   💾 处理摘要已保存: {summary_filename}")
            
        except Exception as e:
            print(f"   ⚠️  创建摘要失败: {e}")
    
    def print_test_summary(self, total_images, total_detections, corrections, high_conf_ocr, results_dir):
        """输出测试摘要"""
        print(f"\n📋 YOLO + 箭头修正测试摘要")
        print("=" * 100)
        print(f"🖼️  处理图像总数: {total_images}")
        print(f"🎯 检测目标总数: {total_detections}")
        print(f"🔄 成功修正数量: {corrections} ({corrections/max(1,total_detections)*100:.1f}%)")
        print(f"📈 高置信度OCR: {high_conf_ocr} ({high_conf_ocr/max(1,total_detections)*100:.1f}%)")
        print(f"📁 结果保存目录: {results_dir}/")
        
        # 创建详细报告
        self.create_comprehensive_report(total_images, total_detections, corrections, high_conf_ocr, results_dir)
    
    def create_comprehensive_report(self, total_images, total_detections, corrections, high_conf_ocr, results_dir):
        """创建综合测试报告"""
        report_path = os.path.join(results_dir, "comprehensive_test_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLO + 箭头方向修正综合测试报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 测试统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"处理图像总数: {total_images}\n")
            f.write(f"检测目标总数: {total_detections}\n")
            f.write(f"成功修正数量: {corrections}\n")
            f.write(f"高置信度OCR: {high_conf_ocr}\n")
            f.write(f"修正成功率: {corrections/max(1,total_detections)*100:.1f}%\n")
            f.write(f"OCR成功率: {high_conf_ocr/max(1,total_detections)*100:.1f}%\n\n")
            
            f.write("🔧 技术方案\n")
            f.write("-" * 40 + "\n")
            f.write("1. YOLO目标检测: 使用best1.pt模型，置信度阈值0.25\n")
            f.write("2. 箭头方向检测: HSV色彩空间 + 凸包缺陷分析\n")
            f.write("3. 智能旋转修正: 四方向测试 + OCR验证\n")
            f.write("4. 质量保证: 高精度仿射变换 + 边界自适应\n\n")
            
            f.write("💡 算法优势\n")
            f.write("-" * 40 + "\n")
            f.write("- 端到端处理: 从完整图像到目标识别修正\n")
            f.write("- 高精度检测: YOLO + 箭头方向双重验证\n")
            f.write("- 智能优化: OCR结果驱动的最优旋转选择\n")
            f.write("- 鲁棒性强: 多种异常情况的优雅处理\n")
        
        print(f"📄 综合测试报告已保存: {report_path}")

def main():
    """主函数"""
    print("🚀 启动YOLO + 箭头方向修正综合测试")
    
    # 创建测试器
    tester = YOLOArrowTester()
    
    # 运行测试
    tester.test_manual_images_with_yolo()
    
    print("\n✅ 测试完成!")

if __name__ == "__main__":
    main() 