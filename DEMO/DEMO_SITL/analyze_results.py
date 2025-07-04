#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果文件分析脚本
分析 raw_detections.json, dual_thread_results.json, median_coordinates.json
"""

import json
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_json_file(filename):
    """加载JSON文件"""
    if not os.path.exists(filename):
        print(f"❌ 文件不存在: {filename}")
        return None
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载: {filename}")
        return data
    except Exception as e:
        print(f"❌ 加载失败 {filename}: {e}")
        return None

def analyze_raw_detections(data):
    """分析原始检测数据"""
    print("\n" + "="*60)
    print("📊 原始检测数据分析 (raw_detections.json)")
    print("="*60)
    
    if not data:
        return
    
    # 基本统计
    total_detections = len(data)
    print(f"🎯 检测总数: {total_detections}")
    
    # 按帧统计
    frame_counts = Counter(item['frame_id'] for item in data)
    max_frame = max(frame_counts.keys())
    min_frame = min(frame_counts.keys())
    print(f"📹 处理帧数: {min_frame} - {max_frame} (共 {max_frame - min_frame + 1} 帧)")
    print(f"📈 平均每帧检测数: {total_detections / len(frame_counts):.2f}")
    
    # 置信度统计
    confidences = [item['confidence'] for item in data]
    print(f"🎯 置信度统计:")
    print(f"   最高: {max(confidences):.4f}")
    print(f"   最低: {min(confidences):.4f}")
    print(f"   平均: {np.mean(confidences):.4f}")
    print(f"   中位数: {np.median(confidences):.4f}")
    
    # 时间统计
    timestamps = [item['timestamp'] for item in data]
    duration = max(timestamps) - min(timestamps)
    print(f"⏱️ 处理时长: {duration:.2f} 秒")
    print(f"🔄 检测频率: {total_detections / duration:.2f} 次/秒")
    
    # 检测框大小统计
    box_areas = []
    for item in data:
        box = item['detection_box']
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        box_areas.append(area)
    
    print(f"📦 检测框面积统计:")
    print(f"   最大: {max(box_areas)} 像素²")
    print(f"   最小: {min(box_areas)} 像素²")
    print(f"   平均: {np.mean(box_areas):.0f} 像素²")
    
    return {
        'total_detections': total_detections,
        'frame_count': len(frame_counts),
        'duration': duration,
        'avg_confidence': np.mean(confidences)
    }

def analyze_dual_thread_results(data):
    """分析副线程处理结果"""
    print("\n" + "="*60)
    print("🔄 副线程处理结果分析 (dual_thread_results.json)")
    print("="*60)
    
    if not data:
        return
    
    total_processed = len(data)
    print(f"⚙️ 副线程处理总数: {total_processed}")
    
    # OCR识别统计
    ocr_results = [item['detected_number'] for item in data]
    recognized_count = sum(1 for result in ocr_results if result != "未识别")
    unrecognized_count = total_processed - recognized_count
    
    print(f"🔍 OCR识别统计:")
    print(f"   成功识别: {recognized_count} ({recognized_count/total_processed*100:.1f}%)")
    print(f"   未识别: {unrecognized_count} ({unrecognized_count/total_processed*100:.1f}%)")
    
    # 识别的数字统计
    if recognized_count > 0:
        recognized_numbers = [result for result in ocr_results if result != "未识别"]
        number_counts = Counter(recognized_numbers)
        print(f"📊 识别到的数字分布:")
        for number, count in sorted(number_counts.items()):
            print(f"   数字 '{number}': {count} 次")
    
    # GPS坐标范围
    latitudes = [item['gps_position']['latitude'] for item in data]
    longitudes = [item['gps_position']['longitude'] for item in data]
    
    print(f"🗺️ GPS坐标范围:")
    print(f"   纬度: {min(latitudes):.6f} ~ {max(latitudes):.6f}")
    print(f"   经度: {min(longitudes):.6f} ~ {max(longitudes):.6f}")
    print(f"   纬度跨度: {max(latitudes) - min(latitudes):.6f}°")
    print(f"   经度跨度: {max(longitudes) - min(longitudes):.6f}°")
    
    # 置信度统计
    confidences = [item['confidence'] for item in data]
    print(f"🎯 处理目标置信度:")
    print(f"   平均: {np.mean(confidences):.4f}")
    print(f"   最高: {max(confidences):.4f}")
    print(f"   最低: {min(confidences):.4f}")
    
    return {
        'total_processed': total_processed,
        'recognized_count': recognized_count,
        'recognition_rate': recognized_count/total_processed*100,
        'coordinate_range': {
            'lat_min': min(latitudes),
            'lat_max': max(latitudes),
            'lon_min': min(longitudes),
            'lon_max': max(longitudes)
        }
    }

def analyze_median_coordinates(data):
    """分析中位数坐标"""
    print("\n" + "="*60)
    print("📍 中位数坐标分析 (median_coordinates.json)")
    print("="*60)
    
    if not data:
        return
    
    print(f"🎯 目标总数: {data['total_targets']}")
    print(f"📍 中位数坐标:")
    print(f"   纬度: {data['median_latitude']:.6f}°")
    print(f"   经度: {data['median_longitude']:.6f}°")
    
    # 转换时间戳
    calc_time = datetime.fromtimestamp(data['calculation_time'])
    print(f"⏱️ 计算时间: {calc_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return data

def generate_summary_report(raw_stats, dual_stats, median_data):
    """生成综合分析报告"""
    print("\n" + "="*60)
    print("📋 综合分析报告")
    print("="*60)
    
    if raw_stats and dual_stats:
        processing_rate = (dual_stats['total_processed'] / raw_stats['total_detections']) * 100
        print(f"🔄 处理效率:")
        print(f"   原始检测: {raw_stats['total_detections']} 个目标")
        print(f"   副线程处理: {dual_stats['total_processed']} 个目标")
        print(f"   处理率: {processing_rate:.1f}%")
        
        if dual_stats['recognized_count'] > 0:
            overall_recognition_rate = (dual_stats['recognized_count'] / raw_stats['total_detections']) * 100
            print(f"   整体OCR成功率: {overall_recognition_rate:.1f}%")
    
    print(f"\n⏱️ 系统性能:")
    if raw_stats:
        print(f"   检测频率: {raw_stats['total_detections'] / raw_stats['duration']:.1f} 次/秒")
        print(f"   平均置信度: {raw_stats['avg_confidence']:.3f}")
    
    print(f"\n🎯 质量评估:")
    if dual_stats:
        if dual_stats['recognition_rate'] >= 50:
            quality = "优秀"
        elif dual_stats['recognition_rate'] >= 30:
            quality = "良好"
        elif dual_stats['recognition_rate'] >= 10:
            quality = "一般"
        else:
            quality = "需改进"
        print(f"   OCR识别率: {dual_stats['recognition_rate']:.1f}% ({quality})")

def create_visualization(raw_data, dual_data):
    """创建可视化图表"""
    print("\n📊 生成可视化图表...")
    
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('双线程SITL系统结果分析', fontsize=16, fontweight='bold')
        
        # 1. 每帧检测数量
        if raw_data:
            frame_counts = Counter(item['frame_id'] for item in raw_data)
            frames = sorted(frame_counts.keys())
            counts = [frame_counts[f] for f in frames]
            
            axes[0,0].plot(frames, counts, 'b-', marker='o', markersize=4)
            axes[0,0].set_title('每帧检测数量')
            axes[0,0].set_xlabel('帧号')
            axes[0,0].set_ylabel('检测数量')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. 置信度分布
        if raw_data:
            confidences = [item['confidence'] for item in raw_data]
            axes[0,1].hist(confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0,1].set_title('检测置信度分布')
            axes[0,1].set_xlabel('置信度')
            axes[0,1].set_ylabel('频次')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. OCR识别结果
        if dual_data:
            ocr_results = [item['detected_number'] for item in dual_data]
            recognized = sum(1 for r in ocr_results if r != "未识别")
            unrecognized = len(ocr_results) - recognized
            
            labels = ['已识别', '未识别']
            sizes = [recognized, unrecognized]
            colors = ['#66b3ff', '#ff9999']
            
            axes[1,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title('OCR识别成功率')
        
        # 4. GPS坐标分布
        if dual_data:
            lats = [item['gps_position']['latitude'] for item in dual_data]
            lons = [item['gps_position']['longitude'] for item in dual_data]
            
            axes[1,1].scatter(lons, lats, alpha=0.6, s=30, c='red')
            axes[1,1].set_title('目标GPS坐标分布')
            axes[1,1].set_xlabel('经度')
            axes[1,1].set_ylabel('纬度')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
        print("✅ 图表已保存为: analysis_results.png")
        
    except Exception as e:
        print(f"⚠️ 生成图表时出错: {e}")

def main():
    """主函数"""
    print("🔍 双线程SITL系统结果分析")
    print("=" * 60)
    
    # 加载数据文件
    raw_data = load_json_file('raw_detections.json')
    dual_data = load_json_file('dual_thread_results.json')
    median_data = load_json_file('median_coordinates.json')
    
    # 分析各个文件
    raw_stats = analyze_raw_detections(raw_data)
    dual_stats = analyze_dual_thread_results(dual_data)
    median_stats = analyze_median_coordinates(median_data)
    
    # 生成综合报告
    generate_summary_report(raw_stats, dual_stats, median_data)
    
    # 创建可视化图表
    create_visualization(raw_data, dual_data)
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main() 