#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标数据分析工具
分析无人机打击任务收集的目标数据
"""

import json
import math
import time
from datetime import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class TargetDataAnalyzer:
    """目标数据分析器"""
    
    def __init__(self, data_file="strike_targets.json"):
        """
        初始化分析器
        
        参数:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.targets = []
        self.load_data()
    
    def load_data(self):
        """加载目标数据"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.targets = json.load(f)
            print(f"✅ 成功加载 {len(self.targets)} 个目标数据")
        except FileNotFoundError:
            print(f"❌ 数据文件不存在: {self.data_file}")
            self.targets = []
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            self.targets = []
    
    def analyze_basic_stats(self):
        """基础统计分析"""
        if not self.targets:
            print("⚠️ 没有目标数据可分析")
            return
        
        print("\n📊 基础统计信息")
        print("=" * 50)
        
        # 总体统计
        total_targets = len(self.targets)
        print(f"目标总数: {total_targets}")
        
        # 识别成功率
        recognized_targets = len([t for t in self.targets if t['detected_number'] != "未识别"])
        recognition_rate = recognized_targets / total_targets * 100 if total_targets > 0 else 0
        print(f"识别成功: {recognized_targets} ({recognition_rate:.1f}%)")
        
        # 置信度统计
        confidences = [t['confidence'] for t in self.targets]
        print(f"平均置信度: {np.mean(confidences):.3f}")
        print(f"置信度范围: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
        
        # 识别的数字统计
        numbers = [t['detected_number'] for t in self.targets if t['detected_number'] != "未识别"]
        unique_numbers = list(set(numbers))
        print(f"识别到的不同数字: {len(unique_numbers)}")
        print(f"数字列表: {sorted(unique_numbers)}")
        
        # 时间跨度
        if self.targets:
            timestamps = [t['detection_timestamp'] for t in self.targets]
            time_span = max(timestamps) - min(timestamps)
            print(f"检测时间跨度: {time_span:.1f}秒")
            
            start_time = datetime.fromtimestamp(min(timestamps))
            end_time = datetime.fromtimestamp(max(timestamps))
            print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def analyze_spatial_distribution(self):
        """空间分布分析"""
        if not self.targets:
            return
        
        print("\n🗺️ 空间分布分析")
        print("=" * 50)
        
        # GPS坐标范围
        lats = [t['gps_position']['latitude'] for t in self.targets]
        lons = [t['gps_position']['longitude'] for t in self.targets]
        
        print(f"纬度范围: {min(lats):.6f} - {max(lats):.6f}")
        print(f"经度范围: {min(lons):.6f} - {max(lons):.6f}")
        
        # 计算覆盖区域
        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        
        # 转换为米（近似）
        lat_meters = lat_span * 111000  # 1度纬度约111km
        lon_meters = lon_span * 111000 * math.cos(math.radians(np.mean(lats)))
        
        print(f"覆盖区域: {lat_meters:.0f}m × {lon_meters:.0f}m")
        print(f"总覆盖面积: {lat_meters * lon_meters / 1000000:.2f} 平方公里")
        
        # 像素位置分析
        pixel_xs = [t['pixel_position']['x'] for t in self.targets]
        pixel_ys = [t['pixel_position']['y'] for t in self.targets]
        
        print(f"像素X范围: {min(pixel_xs)} - {max(pixel_xs)}")
        print(f"像素Y范围: {min(pixel_ys)} - {max(pixel_ys)}")
    
    def analyze_flight_data(self):
        """飞行数据分析"""
        if not self.targets:
            return
        
        print("\n✈️ 飞行数据分析")
        print("=" * 50)
        
        # 飞行高度
        altitudes = [t['flight_data']['altitude'] for t in self.targets]
        print(f"飞行高度: {np.mean(altitudes):.1f}m (范围: {min(altitudes):.1f} - {max(altitudes):.1f})")
        
        # 飞行速度
        speeds = [t['flight_data']['ground_speed'] for t in self.targets]
        print(f"地面速度: {np.mean(speeds):.1f}m/s (范围: {min(speeds):.1f} - {max(speeds):.1f})")
        
        # 姿态角度
        pitches = [t['flight_data']['pitch'] for t in self.targets]
        rolls = [t['flight_data']['roll'] for t in self.targets]
        yaws = [t['flight_data']['yaw'] for t in self.targets]
        
        print(f"俯仰角: {np.mean(pitches):.1f}° (范围: {min(pitches):.1f} - {max(pitches):.1f})")
        print(f"横滚角: {np.mean(rolls):.1f}° (范围: {min(rolls):.1f} - {max(rolls):.1f})")
        print(f"偏航角: {np.mean(yaws):.1f}° (范围: {min(yaws):.1f} - {max(yaws):.1f})")
        
        # 航向分析
        headings = [t['flight_data']['heading'] for t in self.targets]
        print(f"航向角: {np.mean(headings):.1f}° (范围: {min(headings):.1f} - {max(headings):.1f})")
    
    def calculate_distances(self):
        """计算目标间距离"""
        if len(self.targets) < 2:
            return
        
        print("\n📏 目标距离分析")
        print("=" * 50)
        
        distances = []
        for i in range(len(self.targets)):
            for j in range(i + 1, len(self.targets)):
                dist = self._calculate_gps_distance(
                    self.targets[i]['gps_position']['latitude'],
                    self.targets[i]['gps_position']['longitude'],
                    self.targets[j]['gps_position']['latitude'],
                    self.targets[j]['gps_position']['longitude']
                )
                distances.append(dist)
        
        if distances:
            print(f"目标间平均距离: {np.mean(distances):.1f}m")
            print(f"最近距离: {min(distances):.1f}m")
            print(f"最远距离: {max(distances):.1f}m")
            print(f"距离标准差: {np.std(distances):.1f}m")
    
    def _calculate_gps_distance(self, lat1, lon1, lat2, lon2):
        """计算两个GPS坐标间的距离（米）"""
        # 使用Haversine公式
        R = 6371000  # 地球半径（米）
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def generate_report(self, output_file="target_analysis_report.txt"):
        """生成分析报告"""
        print(f"\n📄 生成分析报告: {output_file}")
        
        # 重定向输出到文件
        import sys
        original_stdout = sys.stdout
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                sys.stdout = f
                
                print("无人机对地打击任务 - 目标数据分析报告")
                print("=" * 60)
                print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"数据文件: {self.data_file}")
                print()
                
                self.analyze_basic_stats()
                self.analyze_spatial_distribution()
                self.analyze_flight_data()
                self.calculate_distances()
                
                print("\n📋 详细目标列表")
                print("=" * 60)
                for i, target in enumerate(self.targets, 1):
                    print(f"\n目标 {i}: {target['target_id']}")
                    print(f"  识别数字: {target['detected_number']}")
                    print(f"  置信度: {target['confidence']:.3f}")
                    print(f"  GPS坐标: {target['gps_position']['latitude']:.6f}, {target['gps_position']['longitude']:.6f}")
                    print(f"  像素位置: ({target['pixel_position']['x']}, {target['pixel_position']['y']})")
                    print(f"  检测时间: {datetime.fromtimestamp(target['detection_timestamp']).strftime('%H:%M:%S')}")
                    
                    flight = target['flight_data']
                    print(f"  飞行状态: 高度{flight['altitude']:.1f}m, 速度{flight['ground_speed']:.1f}m/s")
                    print(f"  飞机姿态: P{flight['pitch']:.1f}° R{flight['roll']:.1f}° Y{flight['yaw']:.1f}°")
        
        finally:
            sys.stdout = original_stdout
        
        print(f"✅ 报告已保存到: {output_file}")
    
    def plot_target_map(self, save_file="target_map.png"):
        """绘制目标分布地图"""
        if not self.targets:
            print("⚠️ 没有目标数据可绘制")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # 提取坐标
            lats = [t['gps_position']['latitude'] for t in self.targets]
            lons = [t['gps_position']['longitude'] for t in self.targets]
            confidences = [t['confidence'] for t in self.targets]
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            
            # 绘制目标点
            scatter = plt.scatter(lons, lats, c=confidences, s=100, 
                                cmap='viridis', alpha=0.7, edgecolors='black')
            
            # 添加颜色条
            plt.colorbar(scatter, label='检测置信度')
            
            # 添加目标标签
            for i, target in enumerate(self.targets):
                plt.annotate(target['detected_number'], 
                           (target['gps_position']['longitude'], target['gps_position']['latitude']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('经度')
            plt.ylabel('纬度')
            plt.title('无人机打击目标分布图')
            plt.grid(True, alpha=0.3)
            
            # 保存图片
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ 目标分布图已保存: {save_file}")
            
        except ImportError:
            print("❌ 需要安装matplotlib: pip install matplotlib")
        except Exception as e:
            print(f"❌ 绘图失败: {e}")
    
    def export_kml(self, output_file="strike_targets.kml"):
        """导出KML文件（用于Google Earth）"""
        if not self.targets:
            print("⚠️ 没有目标数据可导出")
            return
        
        kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>无人机打击目标</name>
    <description>无人机对地打击任务识别的目标位置</description>
    
    <Style id="target_style">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/target.png</href>
        </Icon>
        <scale>1.2</scale>
      </IconStyle>
    </Style>
"""
        
        for target in self.targets:
            lat = target['gps_position']['latitude']
            lon = target['gps_position']['longitude']
            name = target['target_id']
            number = target['detected_number']
            confidence = target['confidence']
            timestamp = datetime.fromtimestamp(target['detection_timestamp'])
            
            kml_content += f"""
    <Placemark>
      <name>{name}</name>
      <description>
        识别数字: {number}
        置信度: {confidence:.3f}
        检测时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
      </description>
      <styleUrl>#target_style</styleUrl>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>"""
        
        kml_content += """
  </Document>
</kml>"""
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(kml_content)
            print(f"✅ KML文件已导出: {output_file}")
        except Exception as e:
            print(f"❌ KML导出失败: {e}")

def main():
    """主函数"""
    print("📊 目标数据分析工具")
    print("=" * 50)
    
    # 创建分析器
    analyzer = TargetDataAnalyzer("strike_targets.json")
    
    if not analyzer.targets:
        print("⚠️ 没有找到目标数据，请先运行打击任务系统")
        return
    
    # 执行分析
    analyzer.analyze_basic_stats()
    analyzer.analyze_spatial_distribution()
    analyzer.analyze_flight_data()
    analyzer.calculate_distances()
    
    # 生成报告
    analyzer.generate_report()
    
    # 绘制地图
    analyzer.plot_target_map()
    
    # 导出KML
    analyzer.export_kml()
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main() 