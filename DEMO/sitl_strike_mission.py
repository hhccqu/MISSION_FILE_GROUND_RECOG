#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SITL仿真打击任务系统
连接Mission Planner SITL仿真，使用真实MAVLink数据
集成高精度图像转正功能
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strike_mission_system import StrikeMissionSystem
from target_geo_calculator import FlightData, TargetGeoCalculator, TargetDataManager, TargetInfo
from yolo_trt_utils import YOLOTRTDetector
import easyocr
import cv2
import numpy as np
import threading
from queue import Queue, Empty
from typing import Tuple, Optional

# MAVLink相关
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
    print("✅ MAVLink库可用")
except ImportError:
    MAVLINK_AVAILABLE = False
    print("❌ MAVLink库不可用，请安装: pip install pymavlink")

class ImageOrientationCorrector:
    """高精度图像方向校正器"""
    
    def __init__(self, debug_mode: bool = False):
        """
        初始化校正器
        
        Args:
            debug_mode: 是否开启调试模式
        """
        self.debug_mode = debug_mode
        self.debug_images = {}
        self.correction_stats = {
            'total_processed': 0,
            'successful_corrections': 0,
            'failed_corrections': 0
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理：基于红色颜色识别进行二值化、形态学操作
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的二值图像
        """
        # 1. 确保图像是BGR格式
        if len(image.shape) != 3:
            return None
        
        # 2. 高斯滤波去噪
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 3. 基于红色的颜色分割
        red_mask_bgr = self._create_red_mask_bgr(blurred)
        red_mask_hsv = self._create_red_mask_hsv(blurred)
        red_mask_lab = self._create_red_mask_lab(blurred)
        
        # 组合多个颜色空间的结果
        combined_mask = cv2.bitwise_or(red_mask_hsv, red_mask_bgr)
        combined_mask = cv2.bitwise_or(combined_mask, red_mask_lab)
        
        # 4. 形态学操作优化掩码
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 开运算：去除小噪点
        opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        
        # 闭运算：填充小孔洞
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
        
        # 膨胀操作：增强连通性
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
        
        # 红色范围1 (0-10度)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        
        # 红色范围2 (170-180度)
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
            # 1. 预处理
            processed = self.preprocess_image(image)
            if processed is None:
                info['error_message'] = "预处理失败"
                self.correction_stats['failed_corrections'] += 1
                return image, info
            
            # 2. 找到最大轮廓
            contour = self.find_largest_contour(processed)
            if contour is None:
                info['error_message'] = "未找到有效轮廓"
                self.correction_stats['failed_corrections'] += 1
                return image, info
            
            info['contour_area'] = cv2.contourArea(contour)
            
            # 3. 找到尖端点
            tip_point = self.find_tip_point(contour)
            if tip_point is None:
                info['error_message'] = "未找到尖端点"
                self.correction_stats['failed_corrections'] += 1
                return image, info
            
            info['tip_point'] = tip_point
            
            # 4. 计算旋转角度
            image_center = (image.shape[1] // 2, image.shape[0] // 2)
            angle = self.calculate_rotation_angle(tip_point, image_center)
            info['rotation_angle'] = angle
            
            # 5. 旋转图像
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
    
    def __init__(self, connection_string="udpin:localhost:14550"):
        """
        初始化SITL连接
        
        参数:
            connection_string: SITL连接字符串
                - UDP: "udpin:localhost:14550" (Mission Planner默认)
                - TCP: "tcp:localhost:5760" 
                - 串口: "/dev/ttyUSB0:57600"
        """
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
            print("❌ MAVLink库不可用")
            return False
            
        try:
            print(f"🔗 连接SITL仿真: {self.connection_string}")
            
            # 创建MAVLink连接
            self.connection = mavutil.mavlink_connection(
                self.connection_string,
                source_system=255,
                source_component=0
            )
            
            print("⏳ 等待SITL心跳包...")
            heartbeat = self.connection.wait_heartbeat(timeout=10)
            
            if heartbeat:
                print("✅ SITL连接成功!")
                print(f"   系统ID: {self.connection.target_system}")
                print(f"   组件ID: {self.connection.target_component}")
                print(f"   飞控类型: {heartbeat.type}")
                print(f"   自驾仪: {heartbeat.autopilot}")
                
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
            print("📡 请求数据流...")
            
            # 请求位置信息 (5Hz)
            self.connection.mav.request_data_stream_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                5,  # 5Hz
                1   # 启用
            )
            
            # 请求姿态信息 (10Hz)
            self.connection.mav.request_data_stream_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
                10,  # 10Hz
                1    # 启用
            )
            
            print("✅ 数据流请求已发送")
            
        except Exception as e:
            print(f"⚠️ 数据流请求失败: {e}")
    
    def _start_monitoring(self):
        """启动数据监控线程"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 数据监控已启动")
    
    def _monitor_loop(self):
        """数据监控循环"""
        while self.is_running and self.is_connected:
            try:
                # 接收MAVLink消息
                msg = self.connection.recv_match(blocking=True, timeout=1)
                
                if msg is None:
                    continue
                
                self.message_count += 1
                msg_type = msg.get_type()
                
                # 处理不同类型的消息
                if msg_type == 'HEARTBEAT':
                    self.last_heartbeat = time.time()
                    
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self._handle_gps_position(msg)
                    
                elif msg_type == 'ATTITUDE':
                    self._handle_attitude(msg)
                    
                elif msg_type == 'GPS_RAW_INT':
                    self._handle_gps_raw(msg)
                    
            except Exception as e:
                print(f"⚠️ 数据监控错误: {e}")
                time.sleep(1)
    
    def _handle_gps_position(self, msg):
        """处理GPS位置信息"""
        self.gps_count += 1
        
        # 创建飞行数据
        flight_data = FlightData(
            timestamp=time.time(),
            latitude=msg.lat / 1e7,
            longitude=msg.lon / 1e7,
            altitude=msg.alt / 1000.0,
            pitch=0.0,  # 将在attitude消息中更新
            roll=0.0,
            yaw=0.0,
            ground_speed=np.sqrt(msg.vx**2 + msg.vy**2) / 100.0,
            heading=msg.hdg / 100.0 if msg.hdg != 65535 else 0.0
        )
        
        with self.data_lock:
            if self.latest_flight_data:
                # 保留姿态信息
                flight_data.pitch = self.latest_flight_data.pitch
                flight_data.roll = self.latest_flight_data.roll
                flight_data.yaw = self.latest_flight_data.yaw
            
            self.latest_flight_data = flight_data
    
    def _handle_attitude(self, msg):
        """处理姿态信息"""
        if self.latest_flight_data:
            with self.data_lock:
                # 更新姿态角度（弧度转度）
                self.latest_flight_data.pitch = np.degrees(msg.pitch)
                self.latest_flight_data.roll = np.degrees(msg.roll)
                self.latest_flight_data.yaw = np.degrees(msg.yaw)
    
    def _handle_gps_raw(self, msg):
        """处理原始GPS数据"""
        # 可以在这里处理GPS质量信息
        pass
    
    def get_current_flight_data(self) -> FlightData:
        """获取当前飞行数据"""
        with self.data_lock:
            if self.latest_flight_data:
                return self.latest_flight_data
            else:
                # 返回默认数据
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
    
    def get_connection_status(self):
        """获取连接状态"""
        heartbeat_age = time.time() - self.last_heartbeat if self.last_heartbeat > 0 else 999
        
        return {
            'connected': self.is_connected,
            'running': self.is_running,
            'message_count': self.message_count,
            'gps_count': self.gps_count,
            'heartbeat_age': heartbeat_age,
            'connection_string': self.connection_string
        }
    
    def disconnect(self):
        """断开连接"""
        print("🔌 断开SITL连接...")
        self.is_running = False
        self.is_connected = False
        
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        if self.connection:
            self.connection.close()
        
        print("✅ SITL连接已断开")

class SITLStrikeMissionSystem(StrikeMissionSystem):
    """SITL打击任务系统（集成高精度图像转正）"""
    
    def __init__(self, config=None, sitl_connection="udpin:localhost:14550"):
        """
        初始化SITL打击任务系统
        
        参数:
            config: 配置字典
            sitl_connection: SITL连接字符串
        """
        super().__init__(config)
        self.sitl_connection = sitl_connection
        self.flight_data_provider = None
        self.orientation_corrector = None
        
    def initialize(self):
        """初始化所有组件"""
        print("🚀 初始化SITL打击任务系统...")
        
        # 1. 尝试初始化SITL连接
        print("🛩️ 初始化SITL连接...")
        self.flight_data_provider = SITLFlightDataProvider(self.sitl_connection)
        
        sitl_connected = False
        try:
            sitl_connected = self.flight_data_provider.connect()
        except Exception as e:
            print(f"⚠️ SITL连接失败，将使用模拟数据: {e}")
        
        if not sitl_connected:
            print("⚠️ SITL连接失败，将使用模拟飞行数据继续运行")
            self.flight_data_provider = None
        
        # 2. 初始化高精度图像转正器
        print("🔄 初始化高精度图像转正器...")
        self.orientation_corrector = ImageOrientationCorrector(debug_mode=False)
        
        # 3. 初始化其他组件（与父类相同，但跳过GPS模拟器）
        print("📡 初始化目标检测器...")
        model_path = self._find_model()
        if not model_path:
            raise RuntimeError("未找到模型文件")
        
        self.detector = YOLOTRTDetector(
            model_path=model_path, 
            conf_thres=self.config['conf_threshold']
        )
        
        print("🔤 初始化OCR识别器...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        
        print("📍 初始化地理坐标计算器...")
        self.geo_calculator = TargetGeoCalculator(
            camera_fov_h=self.config['camera_fov_h'],
            camera_fov_v=self.config['camera_fov_v']
        )
        
        print("💾 初始化数据管理器...")
        self.data_manager = TargetDataManager(self.config['save_file'])
        
        print("✅ SITL系统初始化完成！")
        
        # 打印连接状态
        if self.flight_data_provider:
            self._print_sitl_status()
        else:
            print("📊 运行模式: 模拟飞行数据模式")
    
    def _rotate_arrow(self, crop_image):
        """
        高精度箭头旋转校正（使用ImageOrientationCorrector）
        
        Args:
            crop_image: 裁剪的箭头图像
            
        Returns:
            校正后的图像
        """
        try:
            if self.orientation_corrector is None:
                # 如果转正器未初始化，使用简化版本
                return self._rotate_arrow_simple(crop_image)
            
            # 使用高精度转正器
            corrected_image, correction_info = self.orientation_corrector.correct_orientation(crop_image)
            
            # 记录转正信息（静默处理）
            if correction_info['success']:
                # 成功转正
                return corrected_image
            else:
                # 转正失败，静默返回原图
                return crop_image
                
        except Exception as e:
            # 静默处理错误，使用简化版本
            return self._rotate_arrow_simple(crop_image)
    
    def _rotate_arrow_simple(self, crop_image):
        """简化版箭头旋转校正（备用方案）"""
        try:
            # 转换为HSV进行红色检测
            hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
            
            # 红色范围
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([30, 255, 255])
            lower_red2 = np.array([150, 30, 30])
            upper_red2 = np.array([179, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大轮廓
                max_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(max_contour)
                (_, _), (w, h), angle = rect
                
                # 角度修正
                if w > h:
                    angle += 90
                
                # 执行旋转
                (h, w) = crop_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(crop_image, M, (w, h), borderValue=(255, 255, 255))
                
                return rotated
            
        except Exception as e:
            print(f"简化转正失败: {e}")
        
        return crop_image
    
    def process_frame(self, frame):
        """
        处理单帧图像（使用SITL飞行数据或模拟数据）
        """
        self.frame_count += 1
        current_time = time.time()
        
        # 获取飞行数据（SITL或模拟）
        if self.flight_data_provider:
            flight_data = self.flight_data_provider.get_current_flight_data()
        else:
            # 使用模拟飞行数据
            flight_data = FlightData(
                timestamp=current_time,
                latitude=39.7392 + (self.frame_count * 0.0001),  # 模拟移动
                longitude=116.4074 + (self.frame_count * 0.0001),
                altitude=100.0,
                pitch=0.0,
                roll=0.0,
                yaw=45.0,
                ground_speed=15.0,
                heading=45.0
            )
        
        # 其余处理逻辑与父类相同
        detections = self.detector.detect(frame)
        self.stats['total_detections'] += len(detections)
        
        height, width = frame.shape[:2]
        self.geo_calculator.image_height = height
        self.geo_calculator.image_width = width
        
        valid_targets = 0
        processed_frame = frame.copy()
        
        max_targets = min(len(detections), self.config['max_targets_per_frame'])
        
        for i, det in enumerate(detections[:max_targets]):
            if det['confidence'] < self.config['min_confidence']:
                continue
                
            x1, y1, x2, y2 = map(int, det['box'])
            
            try:
                expand_ratio = 0.1
                w, h = x2 - x1, y2 - y1
                x1_exp = max(0, x1 - int(w * expand_ratio))
                y1_exp = max(0, y1 - int(h * expand_ratio))
                x2_exp = min(width, x2 + int(w * expand_ratio))
                y2_exp = min(height, y2 + int(h * expand_ratio))
                
                crop = frame[y1_exp:y2_exp, x1_exp:x2_exp].copy()
                if crop.size == 0:
                    continue
                
                rotated = self._rotate_arrow(crop)
                
                # 暂时禁用OCR识别以提高流畅性
                # ocr_text = ""
                # try:
                #     ocr_text = self._perform_ocr(rotated)
                #     if ocr_text:
                #         self.stats['ocr_success'] += 1
                # except Exception as e:
                #     print(f"OCR识别失败: {e}")
                
                # numbers = self.number_extractor.extract_two_digit_numbers(ocr_text)
                # detected_number = numbers[0] if numbers else "未识别"
                detected_number = "未识别"  # 暂时固定为未识别
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # 使用飞行数据计算目标GPS坐标
                target_lat, target_lon = self.geo_calculator.calculate_target_position(
                    center_x, center_y, flight_data
                )
                
                target_info = TargetInfo(
                    target_id=f"SITL_T{self.detection_count:04d}",
                    detected_number=detected_number,
                    pixel_x=center_x,
                    pixel_y=center_y,
                    confidence=det['confidence'],
                    latitude=target_lat,
                    longitude=target_lon,
                    flight_data=flight_data,
                    timestamp=current_time
                )
                
                self.data_manager.add_target(target_info)
                self.detection_count += 1
                valid_targets += 1
                
                self._draw_detection_result(
                    processed_frame, x1, y1, x2, y2, 
                    target_info, rotated
                )
                
            except Exception as e:
                print(f"处理目标 {i} 时出错: {e}")
                continue
        
        # 绘制飞行信息
        self._draw_sitl_flight_info(processed_frame, flight_data)
        
        # 绘制统计信息
        self._draw_statistics(processed_frame, valid_targets)
        
        return processed_frame, valid_targets
    
    def _draw_detection_result(self, frame, x1, y1, x2, y2, target_info, rotated_crop):
        """在图像上绘制检测结果，包括转正图像和OCR结果"""
        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制目标信息
        info_text = f"ID:{target_info.target_id} 数字:{target_info.detected_number}"
        cv2.putText(frame, info_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制GPS坐标
        gps_text = f"GPS:{target_info.latitude:.6f},{target_info.longitude:.6f}"
        cv2.putText(frame, gps_text, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 绘制置信度
        conf_text = f"Conf:{target_info.confidence:.2f}"
        cv2.putText(frame, conf_text, (x1, y2 + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # 显示转正后的图像缩略图
        if rotated_crop is not None and rotated_crop.size > 0:
            # 调整转正图像大小为固定尺寸
            thumbnail_size = 80
            try:
                # 确保图像不为空
                if rotated_crop.shape[0] > 0 and rotated_crop.shape[1] > 0:
                    rotated_resized = cv2.resize(rotated_crop, (thumbnail_size, thumbnail_size))
                    
                    # 计算缩略图位置（在检测框右上角）
                    thumb_x = min(x2 + 5, frame.shape[1] - thumbnail_size)
                    thumb_y = max(y1, thumbnail_size)
                    
                    # 确保缩略图不超出画面边界
                    if thumb_x + thumbnail_size <= frame.shape[1] and thumb_y - thumbnail_size >= 0:
                        # 在主画面上叠加转正后的图像
                        frame[thumb_y-thumbnail_size:thumb_y, thumb_x:thumb_x+thumbnail_size] = rotated_resized
                        
                        # 给缩略图加边框
                        cv2.rectangle(frame, (thumb_x, thumb_y-thumbnail_size), 
                                    (thumb_x+thumbnail_size, thumb_y), (0, 255, 255), 2)
                        
                        # 在缩略图上方添加"转正后"标签
                        cv2.putText(frame, "转正后", (thumb_x, thumb_y-thumbnail_size-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            
            except Exception as e:
                # 静默处理错误，避免影响流畅性
                pass
    
    def _draw_sitl_flight_info(self, frame, flight_data):
        """绘制飞行信息和图像转正统计"""
        # 获取图像转正统计
        correction_stats = self.orientation_corrector.get_stats() if self.orientation_corrector else {
            'total_processed': 0, 'successful_corrections': 0, 'failed_corrections': 0
        }
        
        success_rate = 0
        if correction_stats['total_processed'] > 0:
            success_rate = (correction_stats['successful_corrections'] / correction_stats['total_processed']) * 100
        
        # 根据是否有SITL连接显示不同信息
        if self.flight_data_provider:
            sitl_status = self.flight_data_provider.get_connection_status()
            info_lines = [
                f"🛩️ SITL模式 - {sitl_status['connection_string']}",
                f"GPS: {flight_data.latitude:.6f}, {flight_data.longitude:.6f}",
                f"高度: {flight_data.altitude:.1f}m",
                f"姿态: P{flight_data.pitch:.1f}° R{flight_data.roll:.1f}° Y{flight_data.yaw:.1f}°",
                f"速度: {flight_data.ground_speed:.1f}m/s 航向: {flight_data.heading:.1f}°",
                f"消息: {sitl_status['message_count']} GPS: {sitl_status['gps_count']}",
                f"心跳: {sitl_status['heartbeat_age']:.1f}s前",
                f"🔄 转正统计: {correction_stats['successful_corrections']}/{correction_stats['total_processed']} ({success_rate:.1f}%)"
            ]
        else:
            info_lines = [
                f"🎮 模拟模式 - 使用模拟飞行数据",
                f"GPS: {flight_data.latitude:.6f}, {flight_data.longitude:.6f}",
                f"高度: {flight_data.altitude:.1f}m",
                f"姿态: P{flight_data.pitch:.1f}° R{flight_data.roll:.1f}° Y{flight_data.yaw:.1f}°",
                f"速度: {flight_data.ground_speed:.1f}m/s 航向: {flight_data.heading:.1f}°",
                f"帧数: {self.frame_count}",
                f"🔄 转正统计: {correction_stats['successful_corrections']}/{correction_stats['total_processed']} ({success_rate:.1f}%)"
            ]
        
        for i, line in enumerate(info_lines):
            # 根据内容选择颜色
            if i == 0:
                color = (0, 255, 0) if self.flight_data_provider else (255, 165, 0)  # SITL用绿色，模拟用橙色
            elif "转正统计" in line:
                color = (0, 255, 255)  # 转正统计用黄色
            else:
                color = (255, 255, 255)  # 其他信息用白色
            
            cv2.putText(frame, line, (10, 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _print_sitl_status(self):
        """打印SITL状态"""
        print("\n📊 SITL连接状态:")
        status = self.flight_data_provider.get_connection_status()
        flight_data = self.flight_data_provider.get_current_flight_data()
        
        print(f"   连接: {'✅ 已连接' if status['connected'] else '❌ 未连接'}")
        print(f"   地址: {status['connection_string']}")
        print(f"   消息数: {status['message_count']}")
        print(f"   GPS数: {status['gps_count']}")
        print(f"   心跳: {status['heartbeat_age']:.1f}秒前")
        print(f"   位置: ({flight_data.latitude:.6f}, {flight_data.longitude:.6f})")
        print(f"   高度: {flight_data.altitude:.1f}m")
        print(f"   姿态: P{flight_data.pitch:.1f}° R{flight_data.roll:.1f}° Y{flight_data.yaw:.1f}°")
    
    def _cleanup(self, cap):
        """清理资源"""
        print("\n🔄 正在清理资源...")
        
        # 保存数据
        self._save_data()
        
        # 断开SITL连接
        if self.flight_data_provider:
            self.flight_data_provider.disconnect()
        
        # 释放视频资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 打印最终统计
        print("\n📈 最终统计:")
        print(f"  处理帧数: {self.frame_count}")
        print(f"  检测总数: {self.stats['total_detections']}")
        print(f"  有效目标: {self.data_manager.get_targets_count()}")
        print(f"  OCR成功: {self.stats['ocr_success']}")
        
        # 打印图像转正统计
        if self.orientation_corrector:
            correction_stats = self.orientation_corrector.get_stats()
            success_rate = 0
            if correction_stats['total_processed'] > 0:
                success_rate = (correction_stats['successful_corrections'] / correction_stats['total_processed']) * 100
            
            print(f"  🔄 图像转正统计:")
            print(f"    总处理: {correction_stats['total_processed']}")
            print(f"    转正成功: {correction_stats['successful_corrections']}")
            print(f"    转正失败: {correction_stats['failed_corrections']}")
            print(f"    成功率: {success_rate:.1f}%")
        
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            print(f"  运行时间: {elapsed:.1f}秒")
            print(f"  平均FPS: {self.frame_count / elapsed:.1f}")
        
        print("✅ SITL任务完成!")

def main():
    """主函数"""
    print("🛩️ SITL仿真打击任务系统 (集成高精度图像转正)")
    print("=" * 60)
    
    # SITL连接配置
    sitl_connections = [
        "tcp:localhost:5760",     # ArduPilot SITL默认TCP端口 (用户当前使用)
        "udpin:localhost:14550",  # Mission Planner默认UDP端口
        "udp:localhost:14540",    # 备用UDP端口
    ]
    
    # 任务配置
    config = {
        'conf_threshold': 0.25,
        'camera_fov_h': 60.0,
        'camera_fov_v': 45.0,
        'altitude': 100.0,  # SITL中的高度
        'save_file': 'sitl_targets_with_correction.json',
        'min_confidence': 0.5,
        'ocr_interval': 5,
        'max_targets_per_frame': 5,
        # 图像转正相关配置
        'orientation_correction': True,  # 是否启用高精度转正
        'correction_debug': False,       # 是否开启转正调试模式
    }
    
    # 视频源
    video_source = "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/video2.mp4"
    
    print("📋 SITL配置:")
    print(f"  视频源: {video_source}")
    print(f"  保存文件: {config['save_file']}")
    print(f"  相机视场角: {config['camera_fov_h']}° × {config['camera_fov_v']}°")
    print(f"  🔄 高精度转正: {'✅ 启用' if config['orientation_correction'] else '❌ 禁用'}")
    print(f"  🐛 转正调试: {'✅ 启用' if config['correction_debug'] else '❌ 禁用'}")
    print()
    
    # 尝试不同的SITL连接
    mission = None
    for sitl_conn in sitl_connections:
        try:
            print(f"🔗 尝试SITL连接: {sitl_conn}")
            mission = SITLStrikeMissionSystem(config, sitl_conn)
            mission.initialize()
            break
        except Exception as e:
            print(f"❌ SITL连接失败: {e}")
            continue
    
    if not mission:
        print("❌ 无法连接到任何SITL仿真")
        print("\n💡 请确保:")
        print("   1. Mission Planner SITL仿真正在运行")
        print("   2. SITL输出端口配置正确")
        print("   3. 防火墙允许UDP/TCP连接")
        return
    
    try:
        print(f"\n🎯 开始SITL打击任务（集成高精度图像转正）...")
        print("按键说明:")
        print("  'q' - 退出任务")
        print("  's' - 保存数据")
        print("  'r' - 重置统计")
        print("  'c' - 清空目标数据")
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