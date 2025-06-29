#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SITL仿真打击任务系统
连接Mission Planner SITL仿真，使用真实MAVLink数据
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

# MAVLink相关
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
    print("✅ MAVLink库可用")
except ImportError:
    MAVLINK_AVAILABLE = False
    print("❌ MAVLink库不可用，请安装: pip install pymavlink")

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
    """SITL打击任务系统"""
    
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
        
    def initialize(self):
        """初始化所有组件"""
        print("🚀 初始化SITL打击任务系统...")
        
        # 1. 初始化SITL连接
        print("🛩️ 初始化SITL连接...")
        self.flight_data_provider = SITLFlightDataProvider(self.sitl_connection)
        
        if not self.flight_data_provider.connect():
            raise RuntimeError("无法连接到SITL仿真")
        
        # 2. 初始化其他组件（与父类相同，但跳过GPS模拟器）
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
        self._print_sitl_status()
    
    def process_frame(self, frame):
        """
        处理单帧图像（使用SITL飞行数据）
        """
        self.frame_count += 1
        current_time = time.time()
        
        # 获取SITL飞行数据
        flight_data = self.flight_data_provider.get_current_flight_data()
        
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
                
                ocr_text = ""
                if self.frame_count % self.config['ocr_interval'] == 0:
                    ocr_text = self._perform_ocr(rotated)
                    if ocr_text:
                        self.stats['ocr_success'] += 1
                
                numbers = self.number_extractor.extract_two_digit_numbers(ocr_text)
                detected_number = numbers[0] if numbers else "未识别"
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # 使用SITL飞行数据计算目标GPS坐标
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
        
        # 绘制SITL飞行信息
        self._draw_sitl_flight_info(processed_frame, flight_data)
        
        # 绘制统计信息
        self._draw_statistics(processed_frame, valid_targets)
        
        return processed_frame, valid_targets
    
    def _draw_sitl_flight_info(self, frame, flight_data):
        """绘制SITL飞行信息"""
        sitl_status = self.flight_data_provider.get_connection_status()
        
        info_lines = [
            f"🛩️ SITL模式 - {sitl_status['connection_string']}",
            f"GPS: {flight_data.latitude:.6f}, {flight_data.longitude:.6f}",
            f"高度: {flight_data.altitude:.1f}m",
            f"姿态: P{flight_data.pitch:.1f}° R{flight_data.roll:.1f}° Y{flight_data.yaw:.1f}°",
            f"速度: {flight_data.ground_speed:.1f}m/s 航向: {flight_data.heading:.1f}°",
            f"消息: {sitl_status['message_count']} GPS: {sitl_status['gps_count']}",
            f"心跳: {sitl_status['heartbeat_age']:.1f}s前"
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)  # 第一行用绿色
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
        
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            print(f"  运行时间: {elapsed:.1f}秒")
            print(f"  平均FPS: {self.frame_count / elapsed:.1f}")
        
        print("✅ SITL任务完成!")

def main():
    """主函数"""
    print("🛩️ SITL仿真打击任务系统")
    print("=" * 50)
    
    # SITL连接配置
    sitl_connections = [
        "udpin:localhost:14550",  # Mission Planner默认UDP端口
        "tcp:localhost:5760",     # ArduPilot SITL默认TCP端口
        "udp:localhost:14540",    # 备用UDP端口
    ]
    
    # 任务配置
    config = {
        'conf_threshold': 0.25,
        'camera_fov_h': 60.0,
        'camera_fov_v': 45.0,
        'altitude': 100.0,  # SITL中的高度
        'save_file': 'sitl_targets.json',
        'min_confidence': 0.5,
        'ocr_interval': 5,
        'max_targets_per_frame': 5
    }
    
    # 视频源
    video_source = "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/video2.mp4"
    
    print("📋 SITL配置:")
    print(f"  视频源: {video_source}")
    print(f"  保存文件: {config['save_file']}")
    print(f"  相机视场角: {config['camera_fov_h']}° × {config['camera_fov_v']}°")
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
        print(f"\n🎯 开始SITL打击任务...")
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