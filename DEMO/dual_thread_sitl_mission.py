#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双线程SITL仿真打击任务系统
主线程：实时YOLO检测 + GPS数据收集 + 视频显示
副线程：图像转正 + OCR识别 + GPS坐标计算
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from target_geo_calculator import FlightData, TargetGeoCalculator, TargetInfo
from yolo_trt_utils import YOLOTRTDetector
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

class ImageOrientationCorrector:
    """高精度图像方向校正器"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.correction_stats = {
            'total_processed': 0,
            'successful_corrections': 0,
            'failed_corrections': 0
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理：基于红色颜色识别进行二值化"""
        if len(image.shape) != 3:
            return None
        
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 多颜色空间红色检测
        red_mask_bgr = self._create_red_mask_bgr(blurred)
        red_mask_hsv = self._create_red_mask_hsv(blurred)
        red_mask_lab = self._create_red_mask_lab(blurred)
        
        # 组合多个颜色空间的结果
        combined_mask = cv2.bitwise_or(red_mask_hsv, red_mask_bgr)
        combined_mask = cv2.bitwise_or(combined_mask, red_mask_lab)
        
        # 形态学操作优化掩码
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
        dilated = cv2.dilate(closed, kernel_small, iterations=1)
        
        return dilated
    
    def _create_red_mask_bgr(self, image: np.ndarray) -> np.ndarray:
        """在BGR颜色空间中创建红色掩码"""
        lower_red1 = np.array([0, 0, 100])
        upper_red1 = np.array([80, 80, 255])
        lower_red2 = np.array([0, 0, 150])
        upper_red2 = np.array([100, 100, 255])
        
        mask1 = cv2.inRange(image, lower_red1, upper_red1)
        mask2 = cv2.inRange(image, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)
    
    def _create_red_mask_hsv(self, image: np.ndarray) -> np.ndarray:
        """在HSV颜色空间中创建红色掩码"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)
    
    def _create_red_mask_lab(self, image: np.ndarray) -> np.ndarray:
        """在LAB颜色空间中创建红色掩码"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lower_red = np.array([20, 150, 150])
        upper_red = np.array([255, 255, 255])
        return cv2.inRange(lab, lower_red, upper_red)
    
    def find_largest_contour(self, binary_image: np.ndarray) -> Optional[np.ndarray]:
        """找到最大的轮廓"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < 100:
            return None
        return largest_contour
    
    def find_tip_point(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """找到轮廓的尖端点"""
        if contour is None or len(contour) < 3:
            return None
        
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return None
        
        # 计算质心
        centroid_x = int(moments['m10'] / moments['m00'])
        centroid_y = int(moments['m01'] / moments['m00'])
        
        # 找到距离质心最远的点
        max_distance = 0
        tip_point = None
        
        for point in contour:
            x, y = point[0]
            distance = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
            if distance > max_distance:
                max_distance = distance
                tip_point = (x, y)
        
        return tip_point
    
    def calculate_rotation_angle(self, tip_point: Tuple[int, int], 
                               image_center: Tuple[int, int]) -> float:
        """计算使尖端朝上所需的旋转角度"""
        dx = tip_point[0] - image_center[0]
        dy = tip_point[1] - image_center[1]
        angle_rad = np.arctan2(dx, -dy)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像"""
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = cv2.warpAffine(image, rotation_matrix, 
                                (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        return rotated
    
    def correct_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """主要的方向校正函数"""
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

class SITLFlightDataProvider:
    """SITL飞行数据提供器"""
    
    def __init__(self, connection_string="tcp:localhost:5760"):
        self.connection_string = connection_string
        self.connection = None
        self.is_connected = False
        self.is_running = False
        self.latest_flight_data = None
        self.data_lock = threading.Lock()
        
        # 统计信息
        self.message_count = 0
        self.gps_count = 0
        self.last_heartbeat = 0
        
    def connect(self) -> bool:
        """连接到SITL"""
        if not MAVLINK_AVAILABLE:
            return False
            
        try:
            print(f"🔗 连接SITL仿真: {self.connection_string}")
            
            self.connection = mavutil.mavlink_connection(
                self.connection_string,
                source_system=255,
                source_component=0
            )
            
            print("⏳ 等待SITL心跳包...")
            heartbeat = self.connection.wait_heartbeat(timeout=10)
            
            if heartbeat:
                print("✅ SITL连接成功!")
                self.is_connected = True
                self._request_data_streams()
                self._start_monitoring()
                return True
            else:
                print("❌ 未收到心跳包")
                return False
                
        except Exception as e:
            print(f"❌ SITL连接失败: {e}")
            return False
    
    def _request_data_streams(self):
        """请求数据流"""
        try:
            # 请求位置信息 (5Hz)
            self.connection.mav.request_data_stream_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                5, 1
            )
            
            # 请求姿态信息 (10Hz)
            self.connection.mav.request_data_stream_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
                10, 1
            )
        except Exception as e:
            print(f"⚠️ 数据流请求失败: {e}")
    
    def _start_monitoring(self):
        """启动数据监控线程"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """数据监控循环"""
        while self.is_running and self.is_connected:
            try:
                msg = self.connection.recv_match(blocking=True, timeout=1)
                if msg is None:
                    continue
                
                self.message_count += 1
                msg_type = msg.get_type()
                
                if msg_type == 'HEARTBEAT':
                    self.last_heartbeat = time.time()
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self._handle_gps_position(msg)
                elif msg_type == 'ATTITUDE':
                    self._handle_attitude(msg)
                    
            except Exception as e:
                print(f"⚠️ 数据监控错误: {e}")
                time.sleep(1)
    
    def _handle_gps_position(self, msg):
        """处理GPS位置信息"""
        self.gps_count += 1
        
        flight_data = FlightData(
            timestamp=time.time(),
            latitude=msg.lat / 1e7,
            longitude=msg.lon / 1e7,
            altitude=msg.alt / 1000.0,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            ground_speed=np.sqrt(msg.vx**2 + msg.vy**2) / 100.0,
            heading=msg.hdg / 100.0 if msg.hdg != 65535 else 0.0
        )
        
        with self.data_lock:
            if self.latest_flight_data:
                flight_data.pitch = self.latest_flight_data.pitch
                flight_data.roll = self.latest_flight_data.roll
                flight_data.yaw = self.latest_flight_data.yaw
            
            self.latest_flight_data = flight_data
    
    def _handle_attitude(self, msg):
        """处理姿态信息"""
        if self.latest_flight_data:
            with self.data_lock:
                self.latest_flight_data.pitch = np.degrees(msg.pitch)
                self.latest_flight_data.roll = np.degrees(msg.roll)
                self.latest_flight_data.yaw = np.degrees(msg.yaw)
    
    def get_current_flight_data(self) -> FlightData:
        """获取当前飞行数据"""
        with self.data_lock:
            if self.latest_flight_data:
                return self.latest_flight_data
            else:
                return FlightData(
                    timestamp=time.time(),
                    latitude=0.0,
                    longitude=0.0,
                    altitude=100.0,
                    pitch=0.0,
                    roll=0.0,
                    yaw=0.0,
                    ground_speed=0.0,
                    heading=0.0
                )
    
    def disconnect(self):
        """断开连接"""
        self.is_running = False
        self.is_connected = False
        
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        if self.connection:
            self.connection.close()

class NumberExtractor:
    """数字提取器"""
    def extract_two_digit_numbers(self, text: str) -> List[str]:
        import re
        if not text:
            return []
        numbers = re.findall(r'\b\d{1,2}\b', text)
        return numbers

class DualThreadSITLMission:
    """双线程SITL任务系统"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        
        # 主线程组件
        self.detector = None
        self.flight_data_provider = None
        
        # 副线程组件
        self.orientation_corrector = None
        self.ocr_reader = None
        self.geo_calculator = None
        self.number_extractor = NumberExtractor()
        
        # 线程通信
        self.detection_queue = queue.Queue(maxsize=100)  # 主线程->副线程
        self.result_queue = queue.Queue(maxsize=100)     # 副线程->主线程
        
        # 统计信息
        self.frame_count = 0
        self.detection_count = 0
        self.main_thread_stats = {
            'total_detections': 0,
            'fps': 0,
            'start_time': None
        }
        self.processing_thread_stats = {
            'total_processed': 0,
            'ocr_success': 0,
            'correction_success': 0
        }
        
        # 运行控制
        self.running = False
        self.processing_thread = None
        
        # 数据存储
        self.raw_detections = []  # 主线程收集的原始检测数据
        self.processed_results = []  # 副线程处理后的结果
        
    def _default_config(self):
        """默认配置"""
        return {
            'conf_threshold': 0.25,
            'camera_fov_h': 60.0,
            'camera_fov_v': 45.0,
            'min_confidence': 0.5,
            'max_targets_per_frame': 5,
            'raw_data_file': 'raw_detections.json',
            'final_results_file': 'dual_thread_results.json',
            'median_coordinates_file': 'median_coordinates.json'
        }
    
    def initialize(self):
        """初始化系统"""
        print("🚀 初始化双线程SITL任务系统...")
        
        # 初始化主线程组件
        print("📡 初始化YOLO检测器...")
        model_path = self._find_model()
        if not model_path:
            raise RuntimeError("未找到模型文件")
        
        self.detector = YOLOTRTDetector(
            model_path=model_path, 
            conf_thres=self.config['conf_threshold']
        )
        
        # 初始化SITL连接
        print("🛩️ 初始化SITL连接...")
        self.flight_data_provider = SITLFlightDataProvider()
        
        sitl_connected = False
        try:
            sitl_connected = self.flight_data_provider.connect()
        except Exception as e:
            print(f"⚠️ SITL连接失败，将使用模拟数据: {e}")
        
        if not sitl_connected:
            print("⚠️ 使用模拟飞行数据模式")
            self.flight_data_provider = None
        
        # 初始化副线程组件
        print("🔄 初始化图像转正器...")
        self.orientation_corrector = ImageOrientationCorrector(debug_mode=False)
        
        print("🔤 初始化OCR识别器...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        
        print("📍 初始化地理坐标计算器...")
        self.geo_calculator = TargetGeoCalculator(
            camera_fov_h=self.config['camera_fov_h'],
            camera_fov_v=self.config['camera_fov_v']
        )
        
        print("✅ 双线程系统初始化完成！")
    
    def _find_model(self):
        """查找模型文件"""
        # 直接使用指定的模型路径
        model_path = "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/weights/best1.pt"
        if os.path.exists(model_path):
            return model_path
            
        # 备用路径
        possible_paths = [
            "../weights/best1.pt",
            "../weights/best.pt",
            "weights/best1.pt",
            "weights/best.pt",
            "best.engine",
            "../best.engine", 
            "../../best.engine",
            "../SIMPLE_TEST_ZHUANGZHENG/best.engine",
            "best.pt",
            "../best.pt",
            "../../best.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def start_processing_thread(self):
        """启动副线程"""
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        print("🔄 副线程已启动")
    
    def _processing_loop(self):
        """副线程处理循环"""
        print("🔄 副线程开始处理...")
        
        while self.running:
            try:
                # 从队列获取检测数据包
                package = self.detection_queue.get(timeout=1)
                if package is None:  # 结束信号
                    break
                
                # 处理图像转正
                corrected_image, correction_info = self.orientation_corrector.correct_orientation(package.crop_image)
                
                # 处理OCR识别
                ocr_text = ""
                try:
                    ocr_results = self.ocr_reader.readtext(corrected_image)
                    if ocr_results:
                        ocr_text = ' '.join([result[1] for result in ocr_results])
                        self.processing_thread_stats['ocr_success'] += 1
                except Exception as e:
                    pass
                
                # 提取数字
                numbers = self.number_extractor.extract_two_digit_numbers(ocr_text)
                detected_number = numbers[0] if numbers else "未识别"
                
                # 计算实际GPS坐标
                target_lat, target_lon = self.geo_calculator.calculate_target_position(
                    package.pixel_center[0], package.pixel_center[1], package.flight_data
                )
                
                # 创建最终结果
                result = TargetInfo(
                    target_id=package.target_id,
                    detected_number=detected_number,
                    pixel_x=package.pixel_center[0],
                    pixel_y=package.pixel_center[1],
                    confidence=package.confidence,
                    latitude=target_lat,
                    longitude=target_lon,
                    flight_data=package.flight_data,
                    timestamp=package.timestamp
                )
                
                # 保存处理结果
                self.processed_results.append(result)
                self.processing_thread_stats['total_processed'] += 1
                
                if correction_info['success']:
                    self.processing_thread_stats['correction_success'] += 1
                
                # 将结果放入结果队列（用于实时显示）
                try:
                    self.result_queue.put_nowait({
                        'target_info': result,
                        'correction_info': correction_info,
                        'corrected_image': corrected_image
                    })
                except queue.Full:
                    pass  # 队列满时忽略
                
                self.detection_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"副线程处理错误: {e}")
                continue
        
        print("🔄 副线程处理完成")
    
    def run_video_mission(self, video_source):
        """运行视频任务"""
        print(f"🎯 开始双线程任务，视频源: {video_source}")
        
        # 打开视频源
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {video_source}")
        
        # 设置窗口
        cv2.namedWindow("双线程SITL任务", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("双线程SITL任务", 1280, 720)
        
        self.running = True
        self.main_thread_stats['start_time'] = time.time()
        
        # 启动副线程
        self.start_processing_thread()
        
        print("📋 任务控制:")
        print("  'q' - 退出任务")
        print("  's' - 保存当前数据")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("视频读取结束")
                    break
                
                # 主线程处理：YOLO检测 + 数据收集
                processed_frame = self._main_thread_process(frame)
                
                # 显示结果
                cv2.imshow("双线程SITL任务", processed_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_current_data()
                
        except KeyboardInterrupt:
            print("\n收到中断信号")
        finally:
            self._cleanup(cap)
    
    def _main_thread_process(self, frame):
        """主线程处理：实时YOLO检测和数据收集"""
        self.frame_count += 1
        current_time = time.time()
        
        # 获取飞行数据
        if self.flight_data_provider:
            flight_data = self.flight_data_provider.get_current_flight_data()
        else:
            # 模拟飞行数据
            flight_data = FlightData(
                timestamp=current_time,
                latitude=39.7392 + (self.frame_count * 0.0001),
                longitude=116.4074 + (self.frame_count * 0.0001),
                altitude=100.0,
                pitch=0.0,
                roll=0.0,
                yaw=45.0,
                ground_speed=15.0,
                heading=45.0
            )
        
        # YOLO检测
        detections = self.detector.detect(frame)
        self.main_thread_stats['total_detections'] += len(detections)
        
        height, width = frame.shape[:2]
        self.geo_calculator.image_height = height
        self.geo_calculator.image_width = width
        
        processed_frame = frame.copy()
        current_targets = 0
        
        # 处理检测结果
        max_targets = min(len(detections), self.config['max_targets_per_frame'])
        
        for i, det in enumerate(detections[:max_targets]):
            if det['confidence'] < self.config['min_confidence']:
                continue
                
            x1, y1, x2, y2 = map(int, det['box'])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            try:
                # 扩展检测框并裁剪图像
                expand_ratio = 0.1
                w, h = x2 - x1, y2 - y1
                x1_exp = max(0, x1 - int(w * expand_ratio))
                y1_exp = max(0, y1 - int(h * expand_ratio))
                x2_exp = min(width, x2 + int(w * expand_ratio))
                y2_exp = min(height, y2 + int(h * expand_ratio))
                
                crop = frame[y1_exp:y2_exp, x1_exp:x2_exp].copy()
                if crop.size == 0:
                    continue
                
                # 创建检测数据包
                target_id = f"DT_T{self.detection_count:04d}"
                package = DetectionPackage(
                    frame_id=self.frame_count,
                    timestamp=current_time,
                    crop_image=crop,
                    detection_box=(x1, y1, x2, y2),
                    pixel_center=(center_x, center_y),
                    confidence=det['confidence'],
                    flight_data=flight_data,
                    target_id=target_id
                )
                
                # 保存原始检测数据
                raw_detection = {
                    'target_id': target_id,
                    'frame_id': self.frame_count,
                    'timestamp': current_time,
                    'detection_box': (x1, y1, x2, y2),
                    'pixel_center': (center_x, center_y),
                    'confidence': det['confidence'],
                    'flight_data': asdict(flight_data)
                }
                self.raw_detections.append(raw_detection)
                
                # 发送到副线程处理队列
                try:
                    self.detection_queue.put_nowait(package)
                except queue.Full:
                    print("⚠️ 检测队列已满，跳过当前目标")
                
                # 在主画面上绘制检测结果
                self._draw_main_detection(processed_frame, x1, y1, x2, y2, 
                                        target_id, det['confidence'])
                
                self.detection_count += 1
                current_targets += 1
                
            except Exception as e:
                print(f"处理目标 {i} 时出错: {e}")
                continue
        
        # 绘制实时信息
        self._draw_main_thread_info(processed_frame, flight_data, current_targets)
        
        return processed_frame
    
    def _draw_main_detection(self, frame, x1, y1, x2, y2, target_id, confidence):
        """绘制主线程检测结果"""
        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制目标信息
        info_text = f"ID:{target_id} Conf:{confidence:.2f}"
        cv2.putText(frame, info_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 绘制状态（等待处理）
        cv2.putText(frame, "等待处理...", (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def _draw_main_thread_info(self, frame, flight_data, current_targets):
        """绘制主线程信息"""
        # 计算FPS
        if self.main_thread_stats['start_time']:
            elapsed = time.time() - self.main_thread_stats['start_time']
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.main_thread_stats['fps'] = fps
        else:
            fps = 0
        
        # 飞行信息
        mode_text = "🛩️ SITL模式" if self.flight_data_provider else "🎮 模拟模式"
        info_lines = [
            f"{mode_text}",
            f"GPS: {flight_data.latitude:.6f}, {flight_data.longitude:.6f}",
            f"高度: {flight_data.altitude:.1f}m",
            f"速度: {flight_data.ground_speed:.1f}m/s",
            f"帧数: {self.frame_count} FPS: {fps:.1f}",
            f"当前目标: {current_targets}",
            f"总检测: {self.main_thread_stats['total_detections']}",
            f"队列: {self.detection_queue.qsize()}/{self.detection_queue.maxsize}"
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if self.flight_data_provider else (255, 165, 0)
            if i >= 4:  # 统计信息用白色
                color = (255, 255, 255)
            cv2.putText(frame, line, (10, 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 副线程统计信息
        stats_lines = [
            f"🔄 副线程统计:",
            f"已处理: {self.processing_thread_stats['total_processed']}",
            f"转正成功: {self.processing_thread_stats['correction_success']}",
            f"OCR成功: {self.processing_thread_stats['ocr_success']}"
        ]
        
        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (frame.shape[1] - 250, 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def _save_current_data(self):
        """保存当前数据"""
        try:
            # 保存原始检测数据
            with open(self.config['raw_data_file'], 'w', encoding='utf-8') as f:
                json.dump(self.raw_detections, f, ensure_ascii=False, indent=2)
            
            # 保存处理结果
            processed_data = []
            for result in self.processed_results:
                processed_data.append({
                    'target_id': result.target_id,
                    'detected_number': result.detected_number,
                    'pixel_position': {'x': result.pixel_x, 'y': result.pixel_y},
                    'confidence': result.confidence,
                    'gps_position': {'latitude': result.latitude, 'longitude': result.longitude},
                    'flight_data': asdict(result.flight_data),
                    'detection_timestamp': result.timestamp
                })
            
            with open(self.config['final_results_file'], 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 数据已保存:")
            print(f"   原始检测: {len(self.raw_detections)} 条 -> {self.config['raw_data_file']}")
            print(f"   处理结果: {len(self.processed_results)} 条 -> {self.config['final_results_file']}")
            
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
    
    def _calculate_median_coordinates(self):
        """计算中位数坐标"""
        if not self.processed_results:
            print("⚠️ 没有处理结果，无法计算中位数坐标")
            return
        
        # 提取所有有效的GPS坐标
        valid_coords = []
        for result in self.processed_results:
            if result.latitude != 0 and result.longitude != 0:
                valid_coords.append((result.latitude, result.longitude))
        
        if not valid_coords:
            print("⚠️ 没有有效的GPS坐标")
            return
        
        # 计算中位数
        lats = sorted([coord[0] for coord in valid_coords])
        lons = sorted([coord[1] for coord in valid_coords])
        
        n = len(valid_coords)
        if n % 2 == 0:
            median_lat = (lats[n//2-1] + lats[n//2]) / 2
            median_lon = (lons[n//2-1] + lons[n//2]) / 2
        else:
            median_lat = lats[n//2]
            median_lon = lons[n//2]
        
        median_coords = {
            'median_latitude': median_lat,
            'median_longitude': median_lon,
            'total_targets': len(valid_coords),
            'calculation_time': time.time()
        }
        
        # 保存中位数坐标
        with open(self.config['median_coordinates_file'], 'w', encoding='utf-8') as f:
            json.dump(median_coords, f, ensure_ascii=False, indent=2)
        
        print(f"📍 中位数坐标计算完成:")
        print(f"   纬度: {median_lat:.8f}")
        print(f"   经度: {median_lon:.8f}")
        print(f"   基于 {len(valid_coords)} 个有效目标")
        print(f"   保存到: {self.config['median_coordinates_file']}")
        
        return median_coords
    
    def _cleanup(self, cap):
        """清理资源"""
        print("\n🔄 正在清理资源...")
        
        # 停止主线程
        self.running = False
        
        # 等待副线程处理完成
        print("⏳ 等待副线程完成处理...")
        
        # 发送结束信号给副线程
        try:
            self.detection_queue.put(None, timeout=1)
        except queue.Full:
            pass
        
        # 等待副线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        # 保存最终数据
        self._save_current_data()
        
        # 计算中位数坐标
        median_coords = self._calculate_median_coordinates()
        
        # 断开SITL连接
        if self.flight_data_provider:
            self.flight_data_provider.disconnect()
        
        # 释放视频资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 打印最终统计
        print("\n📈 最终统计:")
        print(f"  🎯 主线程:")
        print(f"    处理帧数: {self.frame_count}")
        print(f"    检测总数: {self.main_thread_stats['total_detections']}")
        print(f"    平均FPS: {self.main_thread_stats['fps']:.1f}")
        
        print(f"  🔄 副线程:")
        print(f"    处理目标: {self.processing_thread_stats['total_processed']}")
        print(f"    转正成功: {self.processing_thread_stats['correction_success']}")
        print(f"    OCR成功: {self.processing_thread_stats['ocr_success']}")
        
        if self.main_thread_stats['start_time']:
            elapsed = time.time() - self.main_thread_stats['start_time']
            print(f"  总运行时间: {elapsed:.1f}秒")
        
        print("✅ 双线程任务完成!")

def main():
    """主函数"""
    print("🛩️ 双线程SITL仿真打击任务系统")
    print("=" * 60)
    
    # 任务配置
    config = {
        'conf_threshold': 0.25,
        'camera_fov_h': 60.0,
        'camera_fov_v': 45.0,
        'min_confidence': 0.5,
        'max_targets_per_frame': 5,
        'raw_data_file': 'raw_detections.json',
        'final_results_file': 'dual_thread_results.json',
        'median_coordinates_file': 'median_coordinates.json'
    }
    
    # 视频源
    video_source = "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/video2.mp4"
    
    print("📋 双线程配置:")
    print(f"  视频源: {video_source}")
    print(f"  原始数据: {config['raw_data_file']}")
    print(f"  处理结果: {config['final_results_file']}")
    print(f"  中位数坐标: {config['median_coordinates_file']}")
    print()
    
    # 创建任务系统
    mission = DualThreadSITLMission(config)
    
    try:
        # 初始化系统
        mission.initialize()
        
        print(f"\n🎯 开始双线程任务...")
        print("线程分工:")
        print("  🎯 主线程: YOLO检测 + GPS收集 + 实时显示")
        print("  🔄 副线程: 图像转正 + OCR识别 + GPS计算")
        print()
        
        mission.run_video_mission(video_source)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断任务")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 