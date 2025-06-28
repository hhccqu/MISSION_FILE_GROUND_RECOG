#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标中位数查找器
分析识别出的数字目标，找到中位数作为打击目标
"""

import json
import statistics
from typing import List, Dict, Tuple, Optional
import re

class TargetMedianFinder:
    """目标中位数查找器"""
    
    def __init__(self, data_file: str = "strike_targets.json"):
        """
        初始化中位数查找器
        
        参数:
            data_file: 目标数据文件路径
        """
        self.data_file = data_file
        self.targets_data = []
        self.valid_targets = []
        
    def load_targets_data(self) -> bool:
        """加载目标数据"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.targets_data = json.load(f)
            print(f"✅ 成功加载 {len(self.targets_data)} 个目标数据")
            return True
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False
    
    def extract_valid_numbers(self) -> List[Dict]:
        """提取所有有效的数字目标"""
        self.valid_targets = []
        
        for target in self.targets_data:
            detected_number = target.get('detected_number', '')
            
            # 检查是否为有效的二位数
            if self._is_valid_two_digit_number(detected_number):
                number_value = int(detected_number)
                
                target_info = {
                    'target_id': target.get('target_id', ''),
                    'number': number_value,
                    'detected_number': detected_number,
                    'confidence': target.get('confidence', 0.0),
                    'gps_position': target.get('gps_position', {}),
                    'pixel_position': target.get('pixel_position', {}),
                    'flight_data': target.get('flight_data', {}),
                    'detection_timestamp': target.get('detection_timestamp', 0.0)
                }
                
                self.valid_targets.append(target_info)
        
        print(f"🎯 找到 {len(self.valid_targets)} 个有效数字目标")
        return self.valid_targets
    
    def _is_valid_two_digit_number(self, text: str) -> bool:
        """检查是否为有效的二位数"""
        if not text or text == "未识别":
            return False
        
        # 检查是否为纯数字且为二位数
        if re.match(r'^\d{2}$', text):
            number = int(text)
            return 10 <= number <= 99
        
        return False
    
    def find_median_target(self) -> Optional[Dict]:
        """找到中位数目标"""
        if not self.valid_targets:
            print("❌ 没有有效的数字目标")
            return None
        
        # 提取所有数字
        numbers = [target['number'] for target in self.valid_targets]
        
        # 计算中位数
        median_value = statistics.median(numbers)
        
        print(f"📊 数字统计:")
        print(f"   总数: {len(numbers)}")
        print(f"   最小值: {min(numbers)}")
        print(f"   最大值: {max(numbers)}")
        print(f"   平均值: {statistics.mean(numbers):.2f}")
        print(f"   中位数: {median_value}")
        
        # 找到最接近中位数的目标
        median_target = self._find_closest_to_median(median_value)
        
        if median_target:
            print(f"\n🎯 中位数目标:")
            print(f"   目标ID: {median_target['target_id']}")
            print(f"   数字: {median_target['number']}")
            print(f"   置信度: {median_target['confidence']:.3f}")
            print(f"   GPS坐标: ({median_target['gps_position']['latitude']:.7f}, {median_target['gps_position']['longitude']:.7f})")
        
        return median_target
    
    def _find_closest_to_median(self, median_value: float) -> Optional[Dict]:
        """找到最接近中位数的目标"""
        if not self.valid_targets:
            return None
        
        # 如果中位数是整数，直接查找该数字
        if median_value == int(median_value):
            target_number = int(median_value)
            
            # 查找该数字的目标，优先选择置信度最高的
            matching_targets = [t for t in self.valid_targets if t['number'] == target_number]
            
            if matching_targets:
                # 按置信度排序，选择最高的
                matching_targets.sort(key=lambda x: x['confidence'], reverse=True)
                return matching_targets[0]
        
        # 如果中位数不是整数，找最接近的数字
        closest_target = min(self.valid_targets, 
                           key=lambda x: abs(x['number'] - median_value))
        
        return closest_target
    
    def get_number_distribution(self) -> Dict[int, int]:
        """获取数字分布统计"""
        if not self.valid_targets:
            return {}
        
        distribution = {}
        for target in self.valid_targets:
            number = target['number']
            distribution[number] = distribution.get(number, 0) + 1
        
        return distribution
    
    def print_distribution(self):
        """打印数字分布"""
        distribution = self.get_number_distribution()
        
        if not distribution:
            print("❌ 没有数字分布数据")
            return
        
        print(f"\n📈 数字分布统计:")
        print("-" * 40)
        
        # 按数字排序
        sorted_numbers = sorted(distribution.keys())
        
        for number in sorted_numbers:
            count = distribution[number]
            percentage = (count / len(self.valid_targets)) * 100
            bar = "█" * min(int(percentage / 2), 20)  # 最大20个字符的条形图
            print(f"   {number:2d}: {count:3d} 次 ({percentage:5.1f}%) {bar}")
    
    def get_high_confidence_targets(self, min_confidence: float = 0.6) -> List[Dict]:
        """获取高置信度目标"""
        if not self.valid_targets:
            return []
        
        high_conf_targets = [t for t in self.valid_targets if t['confidence'] >= min_confidence]
        
        print(f"🔍 置信度 >= {min_confidence:.1f} 的目标: {len(high_conf_targets)} 个")
        
        return high_conf_targets

def main():
    """主函数"""
    print("🎯 目标中位数查找器")
    print("=" * 50)
    
    # 创建查找器
    finder = TargetMedianFinder()
    
    # 加载数据
    if not finder.load_targets_data():
        return
    
    # 提取有效数字目标
    valid_targets = finder.extract_valid_numbers()
    
    if not valid_targets:
        print("❌ 没有找到有效的数字目标")
        return
    
    # 打印数字分布
    finder.print_distribution()
    
    # 找到中位数目标
    median_target = finder.find_median_target()
    
    if median_target:
        print(f"\n✅ 找到中位数目标，准备作为打击目标")
        
        # 显示详细信息
        gps_pos = median_target['gps_position']
        pixel_pos = median_target['pixel_position']
        
        print(f"\n📍 打击目标详细信息:")
        print(f"   目标编号: {median_target['target_id']}")
        print(f"   识别数字: {median_target['detected_number']}")
        print(f"   数值: {median_target['number']}")
        print(f"   检测置信度: {median_target['confidence']:.3f}")
        print(f"   GPS纬度: {gps_pos['latitude']:.7f}°")
        print(f"   GPS经度: {gps_pos['longitude']:.7f}°")
        print(f"   像素位置: ({pixel_pos['x']}, {pixel_pos['y']})")
        
        return median_target
    else:
        print("❌ 未找到中位数目标")
        return None

if __name__ == "__main__":
    main() 