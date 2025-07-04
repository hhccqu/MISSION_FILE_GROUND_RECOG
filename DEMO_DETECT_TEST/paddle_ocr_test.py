#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 数字识别测试
使用yolo_arrow_test_results中的original图片进行测试
"""

import os
import cv2
import json
import time
from datetime import datetime
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np

class PaddleOCRTester:
    def __init__(self, results_dir="DEMO_DETECT_TEST/yolo_arrow_test_results"):
        """
        初始化PaddleOCR测试器
        
        Args:
            results_dir: 测试图片所在目录
        """
        self.results_dir = results_dir
        self.output_dir = "DEMO_DETECT_TEST/paddle_ocr_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化PaddleOCR
        print("🚀 初始化PaddleOCR...")
        self.ocr = PaddleOCR(
            use_textline_orientation=True,  # 启用方向分类器，自动处理旋转
            lang='en'                       # 英文识别
        )
        print("✅ PaddleOCR初始化完成")
        
        self.test_results = []
        
    def get_original_images(self):
        """获取所有original图片路径"""
        image_files = []
        for file in os.listdir(self.results_dir):
            if file.endswith('_original.jpg'):
                image_files.append(os.path.join(self.results_dir, file))
        return sorted(image_files)
    
    def preprocess_image(self, image_path):
        """
        图像预处理
        
        Args:
            image_path: 图像路径
            
        Returns:
            processed_image: 预处理后的图像
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 高斯滤波降噪
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def test_single_image(self, image_path):
        """
        测试单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            result_dict: 测试结果字典
        """
        print(f"📸 测试图片: {os.path.basename(image_path)}")
        
        # 读取原始图像
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"❌ 无法读取图片: {image_path}")
            return None
        
        # 预处理图像
        processed_img = self.preprocess_image(image_path)
        
        start_time = time.time()
        
        # 使用PaddleOCR进行识别
        try:
            # 对原始图像进行OCR
            ocr_results = self.ocr.predict(image_path)
            
            # 对预处理图像进行OCR（转换为3通道）
            processed_3ch = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            processed_results = self.ocr.predict(processed_3ch)
        except Exception as e:
            print(f"❌ OCR识别失败: {str(e)}")
            return None
        
        processing_time = time.time() - start_time
        
        # 解析识别结果
        def parse_ocr_results(results):
            parsed = []
            if results and len(results) > 0:
                # 新版PaddleOCR返回OCRResult对象，可以字典方式访问
                ocr_result = results[0]
                if 'rec_texts' in ocr_result and ocr_result['rec_texts']:
                    texts = ocr_result['rec_texts']
                    scores = ocr_result.get('rec_scores', [])
                    polys = ocr_result.get('rec_polys', [])
                    
                    for i in range(len(texts)):
                        text = texts[i] if i < len(texts) else ""
                        score = scores[i] if i < len(scores) else 0.0
                        poly = polys[i] if i < len(polys) else []
                        
                        parsed.append({
                            'bbox': poly.tolist() if hasattr(poly, 'tolist') else poly,
                            'text': text,
                            'confidence': score
                        })
            return parsed
        
        original_parsed = parse_ocr_results(ocr_results)
        processed_parsed = parse_ocr_results(processed_results)
        
        # 筛选数字结果
        def filter_digits(results):
            digit_results = []
            for item in results:
                text = item['text'].strip()
                # 检查是否包含数字
                if any(c.isdigit() for c in text):
                    # 提取数字部分
                    digits = ''.join(c for c in text if c.isdigit())
                    if len(digits) >= 1:  # 至少包含1个数字
                        item['digits'] = digits
                        digit_results.append(item)
            return digit_results
        
        original_digits = filter_digits(original_parsed)
        processed_digits = filter_digits(processed_parsed)
        
        # 选择最佳结果
        best_result = None
        if processed_digits:
            # 优先选择预处理后的结果
            best_result = max(processed_digits, key=lambda x: x['confidence'])
            best_source = "预处理图像"
        elif original_digits:
            # 备选原始图像结果
            best_result = max(original_digits, key=lambda x: x['confidence'])
            best_source = "原始图像"
        else:
            best_source = "无识别结果"
        
        result = {
            'image_name': os.path.basename(image_path),
            'image_path': image_path,
            'processing_time': processing_time,
            'original_results': original_parsed,
            'processed_results': processed_parsed,
            'original_digits': original_digits,
            'processed_digits': processed_digits,
            'best_result': best_result,
            'best_source': best_source,
            'success': best_result is not None
        }
        
        # 打印结果
        if best_result:
            print(f"✅ 识别成功: '{best_result['digits']}' (置信度: {best_result['confidence']:.3f}) - {best_source}")
        else:
            print("❌ 未识别到数字")
        
        print(f"⏱️  处理时间: {processing_time:.3f}秒")
        print("-" * 50)
        
        return result
    
    def visualize_results(self, result):
        """
        可视化识别结果
        
        Args:
            result: 测试结果字典
        """
        if not result:
            return
        
        img = cv2.imread(result['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始图像结果
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"原始图像 - {result['image_name']}")
        axes[0].axis('off')
        
        # 绘制原始图像的识别框
        for item in result['original_digits']:
            bbox = item['bbox']
            # 创建矩形框
            rect = patches.Polygon(bbox, linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            # 添加文本标签
            x, y = bbox[0]
            axes[0].text(x, y-10, f"{item['digits']} ({item['confidence']:.2f})", 
                        color='red', fontsize=10, fontweight='bold')
        
        # 预处理图像结果
        processed_img = self.preprocess_image(result['image_path'])
        if processed_img is not None:
            axes[1].imshow(processed_img, cmap='gray')
            axes[1].set_title("预处理图像")
            axes[1].axis('off')
            
            # 绘制预处理图像的识别框
            for item in result['processed_digits']:
                bbox = item['bbox']
                rect = patches.Polygon(bbox, linewidth=2, edgecolor='blue', facecolor='none')
                axes[1].add_patch(rect)
                x, y = bbox[0]
                axes[1].text(x, y-10, f"{item['digits']} ({item['confidence']:.2f})", 
                            color='blue', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存可视化结果
        output_path = os.path.join(self.output_dir, f"{result['image_name']}_paddle_result.jpg")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 可视化结果已保存: {output_path}")
    
    def run_test(self):
        """运行完整测试"""
        print("🔥 开始PaddleOCR测试")
        print("=" * 60)
        
        # 获取所有original图片
        image_files = self.get_original_images()
        print(f"📁 找到 {len(image_files)} 张测试图片")
        print()
        
        total_start_time = time.time()
        
        # 逐个测试图片
        for image_path in image_files:
            result = self.test_single_image(image_path)
            if result:
                self.test_results.append(result)
                # 生成可视化结果
                self.visualize_results(result)
        
        total_time = time.time() - total_start_time
        
        # 生成测试报告
        self.generate_report(total_time)
    
    def generate_report(self, total_time):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("📊 PaddleOCR测试报告")
        print("=" * 60)
        
        total_images = len(self.test_results)
        successful_images = sum(1 for r in self.test_results if r['success'])
        success_rate = (successful_images / total_images * 100) if total_images > 0 else 0
        
        total_digits_found = sum(len(r['original_digits']) + len(r['processed_digits']) 
                               for r in self.test_results)
        avg_processing_time = sum(r['processing_time'] for r in self.test_results) / total_images if total_images > 0 else 0
        
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"平均处理时间: {avg_processing_time:.3f}秒/图")
        print()
        print(f"📈 测试统计:")
        print(f"  - 测试图片总数: {total_images}")
        print(f"  - 成功识别图片: {successful_images}")
        print(f"  - 识别成功率: {success_rate:.1f}%")
        print(f"  - 发现数字总数: {total_digits_found}")
        print()
        
        # 详细结果
        print("📋 详细结果:")
        for i, result in enumerate(self.test_results, 1):
            status = "✅" if result['success'] else "❌"
            best_text = result['best_result']['digits'] if result['best_result'] else "无"
            confidence = result['best_result']['confidence'] if result['best_result'] else 0
            print(f"  {i:2d}. {status} {result['image_name']:<25} | 识别: {best_text:<5} | 置信度: {confidence:.3f}")
        
        # 保存JSON报告
        report_data = {
            'test_time': datetime.now().isoformat(),
            'total_time': total_time,
            'statistics': {
                'total_images': total_images,
                'successful_images': successful_images,
                'success_rate': success_rate,
                'total_digits_found': total_digits_found,
                'avg_processing_time': avg_processing_time
            },
            'results': self.test_results
        }
        
        report_path = os.path.join(self.output_dir, 'paddle_ocr_test_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细报告已保存: {report_path}")
        print(f"📸 可视化结果保存在: {self.output_dir}")

def main():
    """主函数"""
    try:
        # 创建测试器
        tester = PaddleOCRTester()
        
        # 运行测试
        tester.run_test()
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 