#!/usr/bin/env python3
# performance_test.py
# 性能对比测试脚本

import cv2
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import threading

# 添加路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'LATEST_CODE'))

try:
    from yolo_trt_utils_optimized import JetsonOptimizedYOLODetector
    from yolo_trt_utils import YOLOTRTDetector  # 原版检测器
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保相关模块在正确路径中")
    sys.exit(1)

# 尝试导入Jetson监控
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

class PerformanceTester:
    """性能测试器"""
    
    def __init__(self):
        self.test_images = []
        self.results = {}
        self.jetson_stats = {}
        
    def generate_test_images(self, count=100, size=(640, 640)):
        """生成测试图像"""
        print(f"📸 生成 {count} 张测试图像...")
        self.test_images = []
        
        for i in range(count):
            # 生成随机图像
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            
            # 添加一些几何形状以模拟真实场景
            cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2)
            cv2.circle(img, (300, 300), 50, (255, 0, 0), -1)
            cv2.putText(img, f"TEST{i}", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            self.test_images.append(img)
        
        print(f"✅ 测试图像生成完成")
    
    def load_real_images(self, image_dir):
        """加载真实图像"""
        if not os.path.exists(image_dir):
            print(f"⚠️  图像目录不存在: {image_dir}")
            return
        
        print(f"📁 从 {image_dir} 加载真实图像...")
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in image_files[:50]:  # 最多加载50张
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                # 调整大小到标准尺寸
                img = cv2.resize(img, (640, 640))
                self.test_images.append(img)
        
        print(f"✅ 加载了 {len(self.test_images)} 张真实图像")
    
    def test_detector_performance(self, detector, detector_name, warmup_runs=5, test_runs=100):
        """测试检测器性能"""
        print(f"\n🔧 测试 {detector_name} 性能...")
        
        if not self.test_images:
            print("❌ 没有测试图像")
            return None
        
        # 预热
        print(f"🔥 预热 {warmup_runs} 次...")
        for i in range(warmup_runs):
            detector.detect(self.test_images[i % len(self.test_images)])
        
        # 性能测试
        print(f"⏱️  执行 {test_runs} 次推理测试...")
        inference_times = []
        detection_counts = []
        
        start_time = time.time()
        
        for i in range(test_runs):
            img = self.test_images[i % len(self.test_images)]
            
            # 单次推理计时
            single_start = time.time()
            detections = detector.detect(img)
            single_time = time.time() - single_start
            
            inference_times.append(single_time)
            detection_counts.append(len(detections))
            
            if (i + 1) % 20 == 0:
                print(f"   完成 {i + 1}/{test_runs} 次测试...")
        
        total_time = time.time() - start_time
        
        # 计算统计信息
        avg_inference_time = sum(inference_times) / len(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        fps = 1.0 / avg_inference_time
        total_fps = test_runs / total_time
        avg_detections = sum(detection_counts) / len(detection_counts)
        
        results = {
            'detector_name': detector_name,
            'total_time': total_time,
            'avg_inference_time': avg_inference_time,
            'min_inference_time': min_inference_time,
            'max_inference_time': max_inference_time,
            'fps': fps,
            'total_fps': total_fps,
            'avg_detections': avg_detections,
            'inference_times': inference_times,
            'detection_counts': detection_counts
        }
        
        # 获取检测器特定统计
        if hasattr(detector, 'get_performance_stats'):
            detector_stats = detector.get_performance_stats()
            results.update(detector_stats)
        
        self.results[detector_name] = results
        
        print(f"✅ {detector_name} 测试完成:")
        print(f"   - 平均推理时间: {avg_inference_time*1000:.2f}ms")
        print(f"   - 平均FPS: {fps:.2f}")
        print(f"   - 总体FPS: {total_fps:.2f}")
        print(f"   - 平均检测数量: {avg_detections:.1f}")
        
        return results
    
    def monitor_jetson_during_test(self, duration=60):
        """在测试期间监控Jetson状态"""
        if not JTOP_AVAILABLE:
            return
        
        print(f"📊 开始监控Jetson状态 ({duration}秒)...")
        
        def monitor():
            stats_history = []
            start_time = time.time()
            
            try:
                with jtop() as jetson:
                    while time.time() - start_time < duration:
                        if jetson.ok():
                            stats = {
                                'timestamp': time.time() - start_time,
                                'gpu_usage': jetson.gpu.get('GR3D', {}).get('val', 0),
                                'cpu_usage': sum(jetson.cpu.values()) / len(jetson.cpu),
                                'memory_used': jetson.memory['RAM']['used'],
                                'memory_total': jetson.memory['RAM']['tot'],
                                'temperature': jetson.temperature.get('CPU', 0),
                                'power': jetson.power.get('cur', 0)
                            }
                            stats_history.append(stats)
                        
                        time.sleep(1)
                
                self.jetson_stats = stats_history
                
            except Exception as e:
                print(f"Jetson监控错误: {e}")
        
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread
    
    def compare_detectors(self, detectors):
        """对比多个检测器"""
        print("\n🔍 开始检测器性能对比...")
        
        # 开始Jetson监控
        monitor_thread = None
        if JTOP_AVAILABLE:
            monitor_thread = self.monitor_jetson_during_test(duration=len(detectors) * 120)
        
        # 测试每个检测器
        for detector_name, detector in detectors.items():
            try:
                self.test_detector_performance(detector, detector_name)
            except Exception as e:
                print(f"❌ 测试 {detector_name} 时出错: {e}")
        
        # 等待监控结束
        if monitor_thread:
            monitor_thread.join(timeout=5)
        
        # 生成对比报告
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """生成对比报告"""
        print("\n📊 生成性能对比报告...")
        
        if len(self.results) < 2:
            print("⚠️  需要至少2个检测器的结果才能对比")
            return
        
        # 打印对比表格
        print("\n" + "="*80)
        print("性能对比报告")
        print("="*80)
        
        # 表头
        print(f"{'检测器':<20} {'平均推理时间(ms)':<15} {'FPS':<10} {'总体FPS':<10} {'平均检测数':<10} {'TensorRT':<10}")
        print("-" * 80)
        
        # 数据行
        for name, result in self.results.items():
            using_trt = result.get('using_tensorrt', False)
            print(f"{name:<20} {result['avg_inference_time']*1000:<15.2f} {result['fps']:<10.2f} "
                  f"{result['total_fps']:<10.2f} {result['avg_detections']:<10.1f} {using_trt:<10}")
        
        # 计算性能提升
        if len(self.results) == 2:
            results_list = list(self.results.values())
            baseline = results_list[0]
            optimized = results_list[1]
            
            speedup = baseline['avg_inference_time'] / optimized['avg_inference_time']
            fps_improvement = (optimized['fps'] - baseline['fps']) / baseline['fps'] * 100
            
            print(f"\n🚀 性能提升:")
            print(f"   - 推理速度提升: {speedup:.2f}x")
            print(f"   - FPS提升: {fps_improvement:.1f}%")
        
        # 保存详细结果
        self.save_detailed_results()
        
        # 生成图表
        self.plot_performance_comparison()
    
    def save_detailed_results(self):
        """保存详细结果到文件"""
        import json
        
        results_file = "performance_test_results.json"
        
        # 准备可序列化的数据
        serializable_results = {}
        for name, result in self.results.items():
            serializable_results[name] = {
                k: v for k, v in result.items() 
                if k not in ['inference_times', 'detection_counts']  # 排除大数组
            }
            # 保存统计信息而不是原始数据
            serializable_results[name]['inference_time_std'] = np.std(result['inference_times'])
            serializable_results[name]['detection_count_std'] = np.std(result['detection_counts'])
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': serializable_results,
                'jetson_stats_summary': self._summarize_jetson_stats(),
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细结果已保存到: {results_file}")
    
    def _summarize_jetson_stats(self):
        """汇总Jetson统计信息"""
        if not self.jetson_stats:
            return {}
        
        summary = {}
        for key in ['gpu_usage', 'cpu_usage', 'temperature', 'power']:
            values = [stat[key] for stat in self.jetson_stats if key in stat]
            if values:
                summary[key] = {
                    'avg': sum(values) / len(values),
                    'max': max(values),
                    'min': min(values)
                }
        
        return summary
    
    def plot_performance_comparison(self):
        """绘制性能对比图表"""
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文
            plt.rcParams['axes.unicode_minus'] = False
            
            if len(self.results) < 2:
                return
            
            # 创建子图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('YOLO检测器性能对比', fontsize=16)
            
            names = list(self.results.keys())
            
            # 1. 推理时间对比
            inference_times = [self.results[name]['avg_inference_time'] * 1000 for name in names]
            ax1.bar(names, inference_times, color=['blue', 'red'])
            ax1.set_title('平均推理时间对比')
            ax1.set_ylabel('时间 (ms)')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. FPS对比
            fps_values = [self.results[name]['fps'] for name in names]
            ax2.bar(names, fps_values, color=['green', 'orange'])
            ax2.set_title('FPS对比')
            ax2.set_ylabel('FPS')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. 推理时间分布（如果有详细数据）
            if 'inference_times' in self.results[names[0]]:
                for i, name in enumerate(names):
                    times = np.array(self.results[name]['inference_times']) * 1000
                    ax3.hist(times, bins=30, alpha=0.7, label=name)
                ax3.set_title('推理时间分布')
                ax3.set_xlabel('时间 (ms)')
                ax3.set_ylabel('频次')
                ax3.legend()
            
            # 4. Jetson资源使用情况
            if self.jetson_stats:
                timestamps = [stat['timestamp'] for stat in self.jetson_stats]
                gpu_usage = [stat['gpu_usage'] for stat in self.jetson_stats]
                cpu_usage = [stat['cpu_usage'] for stat in self.jetson_stats]
                
                ax4.plot(timestamps, gpu_usage, label='GPU使用率', color='red')
                ax4.plot(timestamps, cpu_usage, label='CPU使用率', color='blue')
                ax4.set_title('Jetson资源使用情况')
                ax4.set_xlabel('时间 (s)')
                ax4.set_ylabel('使用率 (%)')
                ax4.legend()
            
            plt.tight_layout()
            plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
            print("📈 性能对比图表已保存到: performance_comparison.png")
            
        except ImportError:
            print("⚠️  matplotlib未安装，跳过图表生成")
        except Exception as e:
            print(f"⚠️  图表生成失败: {e}")

def main():
    """主函数"""
    print("🚀 YOLO检测器性能测试工具")
    print("="*50)
    
    # 查找模型文件
    possible_model_dirs = [
        "../weights",
        "weights", 
        "../ready/weights",
        "D:/AirmodelingTeam/CQU_Ground_Recog_Strile_YoloOcr/weights"
    ]
    
    model_dir = None
    for path in possible_model_dirs:
        if os.path.exists(path):
            model_dir = path
            break
    
    if model_dir is None:
        print("❌ 未找到模型目录")
        return
    
    # 准备模型路径
    pt_model_path = os.path.join(model_dir, "best1.pt")
    trt_model_path = os.path.join(model_dir, "best1.engine")
    
    if not os.path.exists(pt_model_path):
        print(f"❌ 未找到PyTorch模型: {pt_model_path}")
        return
    
    # 初始化性能测试器
    tester = PerformanceTester()
    
    # 生成测试图像
    tester.generate_test_images(count=50, size=(640, 640))
    
    # 尝试加载真实图像
    test_image_dirs = ["../datasets/test/images", "test_images", "images"]
    for img_dir in test_image_dirs:
        if os.path.exists(img_dir):
            tester.load_real_images(img_dir)
            break
    
    # 准备检测器
    detectors = {}
    
    # 原版检测器
    try:
        print("🔧 初始化原版检测器...")
        original_detector = YOLOTRTDetector(pt_model_path, conf_thres=0.25, use_trt=False)
        detectors["原版PyTorch"] = original_detector
    except Exception as e:
        print(f"⚠️  原版检测器初始化失败: {e}")
    
    # TensorRT优化检测器
    try:
        print("🔧 初始化TensorRT优化检测器...")
        if os.path.exists(trt_model_path):
            optimized_detector = JetsonOptimizedYOLODetector(trt_model_path, conf_thres=0.25)
        else:
            optimized_detector = JetsonOptimizedYOLODetector(pt_model_path, conf_thres=0.25)
        detectors["TensorRT优化"] = optimized_detector
    except Exception as e:
        print(f"⚠️  TensorRT检测器初始化失败: {e}")
    
    if not detectors:
        print("❌ 没有可用的检测器")
        return
    
    # 执行性能对比
    tester.compare_detectors(detectors)
    
    print("\n🎉 性能测试完成!")

if __name__ == "__main__":
    main() 