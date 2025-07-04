#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多方向OCR识别器
无需矫正箭头方向，直接识别所有方向的二位数
"""

import cv2
import numpy as np
import easyocr
import re
import time
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor
import threading

class MultiDirectionOCR:
    """多方向OCR识别器"""
    
    def __init__(self):
        """初始化"""
        print("🔤 初始化多方向OCR识别器...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        self.executor = ThreadPoolExecutor(max_workers=4)
        print("✅ 多方向OCR识别器初始化完成")
    
    def recognize_all_directions(self, image: np.ndarray) -> Dict:
        """
        对图像的所有方向进行OCR识别
        
        Args:
            image: 输入图像
            
        Returns:
            dict: 包含所有方向识别结果的字典
        """
        start_time = time.time()
        
        # 生成四个方向的图像
        rotated_images = {
            0: image,  # 原始方向
            90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(image, cv2.ROTATE_180),
            270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        }
        
        # 并行OCR识别
        futures = {}
        for angle, rotated_img in rotated_images.items():
            future = self.executor.submit(self._ocr_single_direction, rotated_img, angle)
            futures[angle] = future
        
        # 收集结果
        results = {}
        for angle, future in futures.items():
            try:
                results[angle] = future.result(timeout=5.0)  # 5秒超时
            except Exception as e:
                print(f"⚠️ 角度{angle}度OCR失败: {e}")
                results[angle] = {
                    'text': '',
                    'confidence': 0.0,
                    'numbers': [],
                    'two_digit_numbers': [],
                    'error': str(e)
                }
        
        # 选择最佳结果
        best_result = self._select_best_result(results)
        
        processing_time = time.time() - start_time
        
        return {
            'all_results': results,
            'best_result': best_result,
            'processing_time': processing_time,
            'success': len(best_result['two_digit_numbers']) > 0
        }
    
    def _ocr_single_direction(self, image: np.ndarray, angle: int) -> Dict:
        """
        对单个方向进行OCR识别
        
        Args:
            image: 图像
            angle: 旋转角度
            
        Returns:
            dict: 识别结果
        """
        try:
            # 图像预处理
            processed_image = self._preprocess_image(image)
            
            # OCR识别
            ocr_results = self.ocr_reader.readtext(processed_image, detail=1)
            
            # 解析结果
            all_text = []
            all_confidences = []
            
            for bbox, text, confidence in ocr_results:
                all_text.append(text.strip())
                all_confidences.append(confidence)
            
            combined_text = ' '.join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            # 提取数字
            numbers = self._extract_numbers(combined_text)
            two_digit_numbers = self._filter_two_digit_numbers(numbers)
            
            return {
                'angle': angle,
                'text': combined_text,
                'confidence': avg_confidence,
                'numbers': numbers,
                'two_digit_numbers': two_digit_numbers,
                'raw_results': ocr_results
            }
            
        except Exception as e:
            return {
                'angle': angle,
                'text': '',
                'confidence': 0.0,
                'numbers': [],
                'two_digit_numbers': [],
                'error': str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理优化OCR识别
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 双边滤波去噪
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def _extract_numbers(self, text: str) -> List[str]:
        """
        从文本中提取所有数字
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 数字列表
        """
        # 清理文本
        cleaned_text = re.sub(r'[^\d\s]', ' ', text)
        
        # 提取连续数字
        numbers = re.findall(r'\d+', cleaned_text)
        
        return numbers
    
    def _filter_two_digit_numbers(self, numbers: List[str]) -> List[str]:
        """
        筛选二位数
        
        Args:
            numbers: 数字列表
            
        Returns:
            List[str]: 二位数列表
        """
        two_digit_numbers = []
        
        for num in numbers:
            if len(num) == 2:
                # 直接是二位数
                two_digit_numbers.append(num)
            elif len(num) > 2:
                # 尝试分割多位数为二位数
                for i in range(0, len(num) - 1, 2):
                    if i + 1 < len(num):
                        two_digit = num[i:i+2]
                        if self._is_valid_two_digit(two_digit):
                            two_digit_numbers.append(two_digit)
        
        return two_digit_numbers
    
    def _is_valid_two_digit(self, num_str: str) -> bool:
        """
        验证是否为有效的二位数
        
        Args:
            num_str: 数字字符串
            
        Returns:
            bool: 是否有效
        """
        try:
            num = int(num_str)
            return 10 <= num <= 99
        except:
            return False
    
    def _select_best_result(self, results: Dict) -> Dict:
        """
        从所有方向的结果中选择最佳结果
        
        Args:
            results: 所有方向的识别结果
            
        Returns:
            dict: 最佳结果
        """
        best_result = {
            'angle': 0,
            'text': '',
            'confidence': 0.0,
            'numbers': [],
            'two_digit_numbers': [],
            'score': 0.0
        }
        
        for angle, result in results.items():
            if 'error' in result:
                continue
            
            # 计算综合得分
            score = self._calculate_result_score(result)
            
            if score > best_result['score']:
                best_result = result.copy()
                best_result['score'] = score
        
        return best_result
    
    def _calculate_result_score(self, result: Dict) -> float:
        """
        计算识别结果的综合得分
        
        Args:
            result: 识别结果
            
        Returns:
            float: 综合得分
        """
        score = 0.0
        
        # 置信度权重 (40%)
        confidence_score = result['confidence'] * 0.4
        
        # 二位数数量权重 (40%)
        two_digit_count = len(result['two_digit_numbers'])
        digit_score = min(two_digit_count / 2.0, 1.0) * 0.4  # 最多2个二位数得满分
        
        # 文本长度合理性权重 (20%)
        text_length = len(result['text'].strip())
        length_score = 0.0
        if 1 <= text_length <= 10:  # 合理的文本长度
            length_score = 0.2
        elif text_length > 0:
            length_score = 0.1
        
        score = confidence_score + digit_score + length_score
        
        # 奖励机制：如果有明确的二位数，额外加分
        if result['two_digit_numbers']:
            score += 0.1
        
        return score
    
    def recognize_two_digit_number(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """
        识别图像中的二位数（主要接口）
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (识别的二位数, 置信度, 详细结果)
        """
        result = self.recognize_all_directions(image)
        
        if result['success'] and result['best_result']['two_digit_numbers']:
            detected_number = result['best_result']['two_digit_numbers'][0]  # 取第一个
            confidence = result['best_result']['confidence']
            
            print(f"🎯 识别成功: {detected_number} (置信度: {confidence:.2f}, 角度: {result['best_result']['angle']}°)")
            
            return detected_number, confidence, result
        else:
            print("❌ 未识别到有效的二位数")
            return "未识别", 0.0, result

# 测试函数
def test_multi_direction_ocr():
    """测试多方向OCR识别器"""
    print("🧪 开始测试多方向OCR识别器...")
    
    # 初始化识别器
    ocr = MultiDirectionOCR()
    
    # 测试图像目录
    test_dir = "test_image_manuel"
    results_dir = "multi_direction_test_results"
    
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        return
    
    # 获取测试图像
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.png')]
    image_files.sort()
    
    print(f"📁 找到 {len(image_files)} 个测试图像")
    
    total_success = 0
    total_time = 0.0
    
    for i, filename in enumerate(image_files, 1):
        print(f"\n🖼️ 测试图像 {i}/{len(image_files)}: {filename}")
        print("-" * 60)
        
        filepath = os.path.join(test_dir, filename)
        image = cv2.imread(filepath)
        
        if image is None:
            print(f"❌ 无法加载图像: {filename}")
            continue
        
        # 识别二位数
        detected_number, confidence, detailed_result = ocr.recognize_two_digit_number(image)
        
        if detected_number != "未识别":
            total_success += 1
        
        total_time += detailed_result['processing_time']
        
        # 显示详细结果
        print(f"📊 详细结果:")
        for angle, result in detailed_result['all_results'].items():
            if 'error' not in result:
                print(f"   {angle:3d}°: '{result['text']}' (置信度: {result['confidence']:.2f}) 二位数: {result['two_digit_numbers']}")
            else:
                print(f"   {angle:3d}°: 识别失败")
        
        print(f"⏱️ 处理时间: {detailed_result['processing_time']:.2f}秒")
    
    # 输出测试摘要
    print(f"\n📋 测试摘要")
    print("=" * 80)
    print(f"🖼️ 总测试图像: {len(image_files)}")
    print(f"✅ 成功识别: {total_success} ({total_success/len(image_files)*100:.1f}%)")
    print(f"⏱️ 平均处理时间: {total_time/len(image_files):.2f}秒")
    print(f"📁 结果保存在: {results_dir}/")

if __name__ == "__main__":
    test_multi_direction_ocr() 