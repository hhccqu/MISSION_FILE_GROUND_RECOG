#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双线程SITL仿真打击任务系统 - Jetson优化版本
主线程：实时YOLO检测 + GPS数据收集 + 视频显示
副线程：图像转正 + OCR识别 + GPS坐标计算
针对Jetson Orin Nano优化：内存管理、GPU加速、功耗控制
"""

import time
import sys
import os
import threading
import queue
import json
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from pathlib import Path
import psutil
import gc

# Jetson特定优化导入
try:
    import jetson.inference
    import jetson.utils
    JETSON_INFERENCE_AVAILABLE = True
    print("✅ Jetson Inference库可用")
except ImportError:
    JETSON_INFERENCE_AVAILABLE = False
    print("📝 使用标准推理库")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from target_geo_calculator import FlightData, TargetGeoCalculator, TargetInfo
from yolo_trt_utils_jetson import YOLOTRTDetectorJetson
import easyocr

# MAVLink相关
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
    print("✅ MAVLink库可用")
except ImportError:
    MAVLINK_AVAILABLE = False
    print("❌ MAVLink库不可用，将使用模拟数据")

@dataclass
class DetectionPackage:
    """检测数据包 - 主线程传递给副线程的数据结构"""
    frame_id: int
    timestamp: float
    crop_image: np.ndarray
    detection_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    pixel_center: Tuple[int, int]  # center_x, center_y
    confidence: float
    flight_data: FlightData
    target_id: str

class JetsonSystemMonitor:
    """Jetson系统监控器"""
    
    def __init__(self):
        self.cpu_temps = []
        self.gpu_temps = []
        self.memory_usage = []
        self.power_usage = []
        
    def get_system_stats(self):
        """获取系统状态"""
        try:
            # CPU温度和使用率
            cpu_temp = self._read_thermal_zone('/sys/class/thermal/thermal_zone0/temp')
            cpu_usage = psutil.cpu_percent()
            
            # GPU温度
            gpu_temp = self._read_thermal_zone('/sys/class/thermal/thermal_zone1/temp')
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 功耗（如果可用）
            power = self._read_power_consumption()
            
            stats = {
                'cpu_temp': cpu_temp,
                'cpu_usage': cpu_usage,
                'gpu_temp': gpu_temp,
                'memory_usage': memory_usage,
                'memory_available': memory.available / (1024**3),  # GB
                'power_consumption': power
            }
            
            # 记录历史数据
            self.cpu_temps.append(cpu_temp)
            self.gpu_temps.append(gpu_temp)
            self.memory_usage.append(memory_usage)
            self.power_usage.append(power)
            
            # 保持最近100个记录
            for hist in [self.cpu_temps, self.gpu_temps, self.memory_usage, self.power_usage]:
                if len(hist) > 100:
                    hist.pop(0)
            
            return stats
            
        except Exception as e:
            print(f"获取系统状态失败: {e}")
            return {}
    
    def _read_thermal_zone(self, path):
        """读取温度传感器"""
        try:
            with open(path, 'r') as f:
                temp = int(f.read().strip()) / 1000.0  # 转换为摄氏度
            return temp
        except:
            return 0.0
    
    def _read_power_consumption(self):
        """读取功耗信息"""
        try:
            # Jetson功耗监控路径
            power_paths = [
                '/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input',
                '/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power1_input'
            ]
            
            total_power = 0
            for path in power_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        power = int(f.read().strip()) / 1000.0  # 转换为瓦特
                        total_power += power
            
            return total_power
        except:
            return 0.0
    
    def check_thermal_throttling(self):
        """检查是否发生温度限流"""
        cpu_temp = self._read_thermal_zone('/sys/class/thermal/thermal_zone0/temp')
        gpu_temp = self._read_thermal_zone('/sys/class/thermal/thermal_zone1/temp')
        
        # Jetson Orin Nano温度阈值
        cpu_throttle_temp = 85.0  # CPU限流温度
        gpu_throttle_temp = 85.0  # GPU限流温度
        
        return {
            'cpu_throttling': cpu_temp > cpu_throttle_temp,
            'gpu_throttling': gpu_temp > gpu_throttle_temp,
            'cpu_temp': cpu_temp,
            'gpu_temp': gpu_temp
        }

class ImageOrientationCorrectorJetson:
    """Jetson优化的图像方向校正器"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.correction_stats = {
            'total_processed': 0,
            'successful_corrections': 0,
            'failed_corrections': 0
        }
        
        # Jetson优化设置
        self._optimize_opencv()
    
    def _optimize_opencv(self):
        """优化OpenCV设置"""
        cv2.setUseOptimized(True)
        cv2.setNumThreads(6)  # 使用所有CPU核心
        
        # 启用OpenCV的GPU加速（如果可用）
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("✅ OpenCV CUDA加速可用")
            else:
                print("📝 OpenCV CUDA加速不可用")
        except:
            print("📝 OpenCV CUDA支持未启用")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理：基于红色颜色识别进行二值化（Jetson优化版本）"""
        if len(image.shape) != 3:
            return None
        
        # 使用更小的核进行高斯滤波以提高性能
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # 多颜色空间红色检测（优化版本）
        red_mask_hsv = self._create_red_mask_hsv_optimized(blurred)
        red_mask_bgr = self._create_red_mask_bgr_optimized(blurred)
        
        # 组合掩码
        combined_mask = cv2.bitwise_or(red_mask_hsv, red_mask_bgr)
        
        # 简化的形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def _create_red_mask_hsv_optimized(self, image: np.ndarray) -> np.ndarray:
        """优化的HSV红色掩码创建"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 使用更宽松的阈值以提高检测率
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)
    
    def _create_red_mask_bgr_optimized(self, image: np.ndarray) -> np.ndarray:
        """优化的BGR红色掩码创建"""
        lower_red = np.array([0, 0, 120])
        upper_red = np.array([100, 100, 255])
        return cv2.inRange(image, lower_red, upper_red)
    
    def find_largest_contour(self, binary_image: np.ndarray) -> Optional[np.ndarray]:
        """找到最大的轮廓（优化版本）"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # 快速筛选有效轮廓
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        if not valid_contours:
            return None
        
        return max(valid_contours, key=cv2.contourArea)
    
    def find_tip_point(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """找到轮廓的尖端点（优化版本）"""
        if contour is None or len(contour) < 3:
            return None
        
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return None
        
        # 计算质心
        centroid_x = int(moments['m10'] / moments['m00'])
        centroid_y = int(moments['m01'] / moments['m00'])
        
        # 使用向量化操作找到最远点
        points = contour.reshape(-1, 2)
        distances = np.sqrt(np.sum((points - [centroid_x, centroid_y])**2, axis=1))
        max_idx = np.argmax(distances)
        
        return tuple(points[max_idx])
    
    def calculate_rotation_angle(self, tip_point: Tuple[int, int], 
                               image_center: Tuple[int, int]) -> float:
        """计算使尖端朝上所需的旋转角度"""
        dx = tip_point[0] - image_center[0]
        dy = tip_point[1] - image_center[1]
        angle_rad = np.arctan2(dx, -dy)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像（优化版本）"""
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 使用更快的插值方法
        rotated = cv2.warpAffine(image, rotation_matrix, 
                                (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        return rotated
    
    def correct_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """主要的方向校正函数（Jetson优化版本）"""
        self.correction_stats['total_processed'] += 1
        
        info = {
            'success': False,
            'rotation_angle': 0,
            'tip_point': None,
            'contour_area': 0,
            'error_message': None
        }
        
        try:
            # 预处理
            processed = self.preprocess_image(image)
            if processed is None:
                info['error_message'] = "预处理失败"
                self.correction_stats['failed_corrections'] += 1
                return image, info
            
            # 找到最大轮廓
            contour = self.find_largest_contour(processed)
            if contour is None:
                info['error_message'] = "未找到有效轮廓"
                self.correction_stats['failed_corrections'] += 1
                return image, info
            
            info['contour_area'] = cv2.contourArea(contour)
            
            # 找到尖端点
            tip_point = self.find_tip_point(contour)
            if tip_point is None:
                info['error_message'] = "未找到尖端点"
                self.correction_stats['failed_corrections'] += 1
                return image, info
            
            info['tip_point'] = tip_point
            
            # 计算旋转角度
            image_center = (image.shape[1] // 2, image.shape[0] // 2)
            angle = self.calculate_rotation_angle(tip_point, image_center)
            info['rotation_angle'] = angle
            
            # 旋转图像
            corrected_image = self.rotate_image(image, angle)
            
            info['success'] = True
            self.correction_stats['successful_corrections'] += 1
            
            return corrected_image, info
            
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            info['error_message'] = error_msg
            self.correction_stats['failed_corrections'] += 1
            return image, info
    
    def get_stats(self):
        """获取校正统计信息"""
        return self.correction_stats.copy()

class DualThreadSITLMissionJetson:
    """Jetson优化的双线程SITL任务系统"""
    
    def __init__(self, config=None):
        """初始化Jetson优化的双线程SITL任务系统"""
        self.config = config or self._default_config()
        
        # 核心组件
        self.yolo_detector = None
        self.orientation_corrector = ImageOrientationCorrectorJetson()
        self.geo_calculator = TargetGeoCalculator()
        self.system_monitor = JetsonSystemMonitor()
        
        # 线程和队列
        self.detection_queue = queue.Queue(maxsize=self.config['detection_queue_size'])
        self.result_queue = queue.Queue(maxsize=self.config['result_queue_size'])
        self.processing_thread = None
        self.is_running = False
        
        # 数据存储
        self.raw_detections = []
        self.processing_results = []
        self.processing_display = {}
        self.target_processing_status = {}
        
        # 性能监控
        self.frame_count = 0
        self.detection_count = 0
        self.processing_start_time = time.time()
        
        # Jetson特定优化
        self._setup_jetson_optimizations()
        
        print("🚀 Jetson优化的双线程SITL任务系统初始化完成")
    
    def _default_config(self):
        """Jetson优化的默认配置"""
        return {
            'model_path': 'weights/best1.pt',
            'confidence_threshold': 0.25,
            'detection_queue_size': 300,  # 减少队列大小以节省内存
            'result_queue_size': 150,
            'queue_wait_timeout': 3.0,    # 减少等待时间
            'use_tensorrt': True,
            'max_fps': 30,                # 限制最大FPS
            'memory_cleanup_interval': 50, # 内存清理间隔
            'thermal_check_interval': 100, # 温度检查间隔
            'power_mode': 'balanced'       # 功耗模式：balanced, performance, power_save
        }
    
    def _setup_jetson_optimizations(self):
        """设置Jetson特定优化"""
        # 设置环境变量
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        
        # 设置功耗模式
        self._set_power_mode(self.config['power_mode'])
        
        # 优化内存管理
        self._setup_memory_optimization()
        
        print("⚡ Jetson系统优化设置完成")
    
    def _set_power_mode(self, mode):
        """设置Jetson功耗模式"""
        mode_map = {
            'power_save': 0,   # 7W模式
            'balanced': 1,     # 15W模式
            'performance': 2   # 25W模式
        }
        
        if mode in mode_map:
            try:
                os.system(f'sudo nvpmodel -m {mode_map[mode]}')
                print(f"🔋 设置功耗模式为: {mode}")
            except:
                print(f"⚠️ 无法设置功耗模式: {mode}")
    
    def _setup_memory_optimization(self):
        """设置内存优化"""
        # 启用内存映射优化
        import mmap
        
        # 设置垃圾回收
        gc.set_threshold(700, 10, 10)
        
        print("💾 内存优化设置完成")
    
    def initialize(self):
        """初始化系统组件"""
        print("🔧 初始化Jetson系统组件...")
        
        # 初始化YOLO检测器
        try:
            self.yolo_detector = YOLOTRTDetectorJetson(
                model_path=self.config['model_path'],
                conf_thres=self.config['confidence_threshold'],
                use_trt=self.config['use_tensorrt']
            )
            self.yolo_detector.optimize_for_jetson()
            print("✅ YOLO检测器初始化完成")
        except Exception as e:
            print(f"❌ YOLO检测器初始化失败: {e}")
            return False
        
        # 初始化OCR
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=True)
            print("✅ OCR识别器初始化完成")
        except Exception as e:
            print(f"❌ OCR识别器初始化失败: {e}")
            return False
        
        return True
    
    def start_processing_thread(self):
        """启动副线程处理"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop_jetson, daemon=True)
        self.processing_thread.start()
        print("🔄 Jetson优化副线程处理已启动")
    
    def _processing_loop_jetson(self):
        """Jetson优化的副线程处理循环"""
        cleanup_counter = 0
        thermal_check_counter = 0
        
        while self.is_running:
            try:
                # 获取检测包
                detection_pkg = self.detection_queue.get(timeout=1.0)
                
                # 更新处理状态
                self.target_processing_status[detection_pkg.target_id] = {
                    'status': 'processing',
                    'start_time': time.time(),
                    'stage': 'orientation_correction'
                }
                
                # 图像转正
                corrected_image, correction_info = self.orientation_corrector.correct_orientation(
                    detection_pkg.crop_image
                )
                
                # 更新状态
                self.target_processing_status[detection_pkg.target_id]['stage'] = 'ocr_recognition'
                
                # OCR识别
                ocr_results = []
                detected_number = ""
                
                if correction_info['success']:
                    try:
                        ocr_results = self.ocr_reader.readtext(corrected_image)
                        if ocr_results:
                            detected_number = ''.join([result[1] for result in ocr_results if result[2] > 0.5])
                    except Exception as e:
                        print(f"OCR识别错误: {e}")
                
                # 更新状态
                self.target_processing_status[detection_pkg.target_id]['stage'] = 'coordinate_calculation'
                
                # GPS坐标计算
                target_lat, target_lon = self.geo_calculator.calculate_target_position(
                    detection_pkg.pixel_center[0],
                    detection_pkg.pixel_center[1], 
                    detection_pkg.flight_data
                )
                
                # 创建处理结果
                result = {
                    'target_id': detection_pkg.target_id,
                    'frame_id': detection_pkg.frame_id,
                    'timestamp': detection_pkg.timestamp,
                    'original_image': detection_pkg.crop_image,
                    'corrected_image': corrected_image,
                    'correction_info': correction_info,
                    'detected_number': detected_number,
                    'ocr_confidence': max([r[2] for r in ocr_results]) if ocr_results else 0.0,
                    'target_coordinates': {
                        'latitude': target_lat,
                        'longitude': target_lon
                    },
                    'flight_data': asdict(detection_pkg.flight_data),
                    'detection_confidence': detection_pkg.confidence,
                    'processing_time': time.time() - self.target_processing_status[detection_pkg.target_id]['start_time']
                }
                
                # 添加到结果队列
                try:
                    self.result_queue.put(result, timeout=1.0)
                    self.processing_results.append(result)
                    
                    # 更新处理显示
                    self.processing_display[detection_pkg.target_id] = result
                    
                    # 更新状态
                    self.target_processing_status[detection_pkg.target_id] = {
                        'status': 'completed',
                        'completion_time': time.time(),
                        'stage': 'completed'
                    }
                    
                except queue.Full:
                    print("⚠️ 结果队列已满")
                
                # 定期清理内存
                cleanup_counter += 1
                if cleanup_counter >= self.config['memory_cleanup_interval']:
                    self._cleanup_memory()
                    cleanup_counter = 0
                
                # 定期检查温度
                thermal_check_counter += 1
                if thermal_check_counter >= self.config['thermal_check_interval']:
                    self._check_thermal_throttling()
                    thermal_check_counter = 0
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"副线程处理错误: {e}")
    
    def _cleanup_memory(self):
        """清理内存"""
        # 限制处理结果数量
        if len(self.processing_results) > 200:
            self.processing_results = self.processing_results[-100:]
        
        # 限制原始检测数量
        if len(self.raw_detections) > 500:
            self.raw_detections = self.raw_detections[-300:]
        
        # 清理处理显示
        if len(self.processing_display) > 50:
            # 保留最新的20个
            latest_keys = list(self.processing_display.keys())[-20:]
            self.processing_display = {k: self.processing_display[k] for k in latest_keys}
        
        # 强制垃圾回收
        gc.collect()
    
    def _check_thermal_throttling(self):
        """检查温度限流"""
        thermal_status = self.system_monitor.check_thermal_throttling()
        
        if thermal_status['cpu_throttling'] or thermal_status['gpu_throttling']:
            print(f"🌡️ 温度警告 - CPU: {thermal_status['cpu_temp']:.1f}°C, GPU: {thermal_status['gpu_temp']:.1f}°C")
            
            # 自动降低处理频率
            time.sleep(0.1)
    
    def run_video_mission_jetson(self, video_source):
        """运行Jetson优化的视频任务"""
        print(f"🎥 启动Jetson优化视频任务: {video_source}")
        
        # 初始化视频捕获
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ 无法打开视频源: {video_source}")
            return
        
        # 设置视频参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, self.config['max_fps'])
        
        # 启动副线程
        self.start_processing_thread()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                self._main_thread_process_jetson(frame)
                
                # 显示结果
                display_frame = self._draw_jetson_interface(frame)
                cv2.imshow('Jetson SITL Mission', display_frame)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_current_data()
                elif key == ord('p'):
                    self._show_performance_stats()
                
        except KeyboardInterrupt:
            print("🛑 用户中断")
        finally:
            self._cleanup_jetson(cap)
    
    def _main_thread_process_jetson(self, frame):
        """Jetson优化的主线程处理"""
        self.frame_count += 1
        current_time = time.time()
        
        # 模拟飞行数据
        flight_data = FlightData(
            timestamp=current_time,
            latitude=39.7462 + (self.frame_count * 0.0001),
            longitude=116.4166 + (self.frame_count * 0.0001),
            altitude=500.0,
            pitch=-10.0,
            roll=0.0,
            yaw=90.0,
            ground_speed=30.0,
            heading=90.0
        )
        
        # YOLO检测
        detections = self.yolo_detector.detect(frame)
        
        # 处理每个检测结果
        for detection in detections:
            self.detection_count += 1
            
            x1, y1, x2, y2 = detection['box']
            confidence = detection['confidence']
            
            # 裁剪目标图像
            crop_image = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop_image.size == 0:
                continue
            
            # 计算像素中心
            pixel_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # 创建检测包
            target_id = f"T{self.detection_count:04d}"
            detection_pkg = DetectionPackage(
                frame_id=self.frame_count,
                timestamp=current_time,
                crop_image=crop_image,
                detection_box=(int(x1), int(y1), int(x2), int(y2)),
                pixel_center=pixel_center,
                confidence=confidence,
                flight_data=flight_data,
                target_id=target_id
            )
            
            # 记录原始检测
            self.raw_detections.append({
                'frame_id': self.frame_count,
                'timestamp': current_time,
                'target_id': target_id,
                'detection_box': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': confidence,
                'flight_data': asdict(flight_data)
            })
            
            # 尝试添加到处理队列
            try:
                self.detection_queue.put(detection_pkg, block=False)
                self.target_processing_status[target_id] = {
                    'status': 'queued',
                    'queue_time': current_time,
                    'stage': 'waiting'
                }
            except queue.Full:
                # 队列满时等待
                wait_start = time.time()
                while time.time() - wait_start < self.config['queue_wait_timeout']:
                    try:
                        self.detection_queue.put(detection_pkg, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                else:
                    print(f"⚠️ 目标 {target_id} 等待超时")
    
    def _draw_jetson_interface(self, frame):
        """绘制Jetson优化的界面"""
        display_frame = frame.copy()
        
        # 获取系统状态
        system_stats = self.system_monitor.get_system_stats()
        
        # 绘制系统信息
        info_y = 30
        cv2.putText(display_frame, f"Jetson Orin Nano - Frame: {self.frame_count}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 25
        cv2.putText(display_frame, f"CPU: {system_stats.get('cpu_temp', 0):.1f}C {system_stats.get('cpu_usage', 0):.1f}%", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        info_y += 25
        cv2.putText(display_frame, f"GPU: {system_stats.get('gpu_temp', 0):.1f}C", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        info_y += 25
        cv2.putText(display_frame, f"Memory: {system_stats.get('memory_usage', 0):.1f}%", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        info_y += 25
        cv2.putText(display_frame, f"Power: {system_stats.get('power_consumption', 0):.1f}W", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 绘制性能统计
        perf_stats = self.yolo_detector.get_performance_stats()
        if perf_stats:
            info_y += 25
            cv2.putText(display_frame, f"YOLO FPS: {perf_stats.get('fps', 0):.1f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return display_frame
    
    def _show_performance_stats(self):
        """显示性能统计"""
        print("\n" + "="*50)
        print("🚀 Jetson性能统计")
        print("="*50)
        
        # 系统统计
        system_stats = self.system_monitor.get_system_stats()
        print(f"CPU温度: {system_stats.get('cpu_temp', 0):.1f}°C")
        print(f"GPU温度: {system_stats.get('gpu_temp', 0):.1f}°C")
        print(f"CPU使用率: {system_stats.get('cpu_usage', 0):.1f}%")
        print(f"内存使用率: {system_stats.get('memory_usage', 0):.1f}%")
        print(f"可用内存: {system_stats.get('memory_available', 0):.1f}GB")
        print(f"功耗: {system_stats.get('power_consumption', 0):.1f}W")
        
        # YOLO性能
        perf_stats = self.yolo_detector.get_performance_stats()
        if perf_stats:
            print(f"YOLO平均推理时间: {perf_stats.get('avg_inference_time', 0)*1000:.1f}ms")
            print(f"YOLO FPS: {perf_stats.get('fps', 0):.1f}")
            print(f"使用TensorRT: {'是' if perf_stats.get('using_tensorrt', False) else '否'}")
        
        # 处理统计
        print(f"处理帧数: {self.frame_count}")
        print(f"检测总数: {self.detection_count}")
        print(f"副线程处理: {len(self.processing_results)}")
        
        # 转正统计
        correction_stats = self.orientation_corrector.get_stats()
        print(f"转正成功率: {correction_stats['successful_corrections']}/{correction_stats['total_processed']}")
        
        print("="*50)
    
    def _save_current_data(self):
        """保存当前数据"""
        timestamp = int(time.time())
        
        # 保存原始检测数据
        raw_file = f"raw_detections_jetson_{timestamp}.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(self.raw_detections, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存处理结果
        results_file = f"dual_thread_results_jetson_{timestamp}.json" 
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.processing_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 数据已保存: {raw_file}, {results_file}")
    
    def _cleanup_jetson(self, cap):
        """Jetson系统清理"""
        print("🧹 清理Jetson系统资源...")
        
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        if cap:
            cap.release()
        
        cv2.destroyAllWindows()
        
        # 最终数据保存
        self._save_current_data()
        
        # 显示最终统计
        self._show_performance_stats()
        
        print("✅ Jetson系统清理完成")

def main():
    """主函数"""
    print("🚀 启动Jetson优化的双线程SITL任务系统")
    
    # 创建系统实例
    mission = DualThreadSITLMissionJetson()
    
    # 初始化系统
    if not mission.initialize():
        print("❌ 系统初始化失败")
        return
    
    # 运行视频任务
    video_source = 0  # 使用摄像头，或指定视频文件路径
    mission.run_video_mission_jetson(video_source)

if __name__ == "__main__":
    main() 