#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成投水任务系统
结合目标检测和MAVLink航点管理
功能：
1. 实时目标检测
2. GPS坐标计算
3. 自动航点插入
4. 投水点计算
5. 任务执行监控
"""

import time
import cv2
import json
import threading
from queue import Queue, Empty
from typing import List, Dict, Optional

# 导入现有模块
from mavlink_waypoint_manager import MAVLinkWaypointManager, DropPoint
from strike_mission_system import StrikeMissionSystem
from target_geo_calculator import FlightData, TargetGeoCalculator

class IntegratedDropMission:
    """集成投水任务系统"""
    
    def __init__(self, config: Dict = None):
        """
        初始化集成系统
        
        Args:
            config: 配置参数
        """
        # 默认配置
        self.config = {
            'mavlink_connection': 'udpin:localhost:14550',
            'video_source': 0,  # 摄像头或视频文件
            'conf_threshold': 0.25,
            'auto_drop_enabled': True,
            'min_target_confidence': 0.6,
            'drop_cooldown': 30.0,  # 投水冷却时间(秒)
            'max_targets_per_mission': 10,
            'save_file': 'drop_targets.json'
        }
        
        if config:
            self.config.update(config)
        
        # 系统组件
        self.waypoint_manager = None
        self.strike_system = None
        self.geo_calculator = TargetGeoCalculator()
        
        # 运行状态
        self.is_running = False
        self.detection_thread = None
        self.mission_thread = None
        
        # 目标管理
        self.detected_targets = Queue()
        self.processed_targets = []
        self.last_drop_time = 0
        self.target_counter = 0
        
        # 数据锁
        self.data_lock = threading.Lock()
        
        print("🚁 集成投水任务系统已初始化")
    
    def initialize(self) -> bool:
        """初始化系统组件"""
        try:
            print("🔧 初始化系统组件...")
            
            # 1. 初始化航点管理器
            print("📡 初始化MAVLink航点管理器...")
            self.waypoint_manager = MAVLinkWaypointManager(
                self.config['mavlink_connection']
            )
            
            if not self.waypoint_manager.connect():
                print("❌ MAVLink连接失败")
                return False
            
            # 2. 初始化目标检测系统
            print("🎯 初始化目标检测系统...")
            strike_config = {
                'conf_threshold': self.config['conf_threshold'],
                'save_file': self.config['save_file'],
                'use_real_gps': True  # 使用真实GPS数据
            }
            
            self.strike_system = StrikeMissionSystem(strike_config)
            if not self.strike_system.initialize():
                print("❌ 目标检测系统初始化失败")
                return False
            
            print("✅ 系统组件初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            return False
    
    def start_mission(self, video_source=None) -> bool:
        """启动集成任务"""
        try:
            if not self.waypoint_manager or not self.strike_system:
                print("❌ 系统未初始化")
                return False
            
            video_src = video_source or self.config['video_source']
            print(f"🚀 启动集成投水任务，视频源: {video_src}")
            
            self.is_running = True
            
            # 启动检测线程
            self.detection_thread = threading.Thread(
                target=self._detection_loop,
                args=(video_src,),
                daemon=True
            )
            self.detection_thread.start()
            
            # 启动任务处理线程
            self.mission_thread = threading.Thread(
                target=self._mission_loop,
                daemon=True
            )
            self.mission_thread.start()
            
            print("✅ 集成任务已启动")
            print("\n💡 控制命令:")
            print("   'q' - 退出任务")
            print("   's' - 保存数据")
            print("   'p' - 暂停/恢复自动投水")
            print("   't' - 显示目标统计")
            print("   'm' - 显示任务状态")
            
            return True
            
        except Exception as e:
            print(f"❌ 启动任务失败: {e}")
            return False
    
    def _detection_loop(self, video_source):
        """目标检测循环"""
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print(f"❌ 无法打开视频源: {video_source}")
                return
            
            print("📹 目标检测循环已启动")
            frame_count = 0
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str):  # 视频文件结束
                        print("📹 视频文件播放完毕")
                        break
                    continue
                
                frame_count += 1
                
                # 获取当前飞行数据
                flight_data = self._get_current_flight_data()
                if not flight_data:
                    continue
                
                # 处理帧
                processed_frame = self._process_detection_frame(frame, flight_data)
                
                # 显示结果
                cv2.imshow('集成投水任务系统', processed_frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("📹 用户请求退出")
                    self.is_running = False
                    break
                elif key == ord('s'):
                    self._save_data()
                elif key == ord('p'):
                    self.config['auto_drop_enabled'] = not self.config['auto_drop_enabled']
                    status = "启用" if self.config['auto_drop_enabled'] else "禁用"
                    print(f"🎯 自动投水已{status}")
                elif key == ord('t'):
                    self._print_target_statistics()
                elif key == ord('m'):
                    self._print_mission_status()
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ 检测循环错误: {e}")
        finally:
            self.is_running = False
    
    def _mission_loop(self):
        """任务处理循环"""
        try:
            print("🎯 任务处理循环已启动")
            
            while self.is_running:
                try:
                    # 获取检测到的目标
                    target_info = self.detected_targets.get(timeout=1)
                    
                    # 检查是否满足投水条件
                    if self._should_process_target(target_info):
                        self._process_target_for_drop(target_info)
                    
                except Empty:
                    continue
                except Exception as e:
                    print(f"⚠️ 任务处理错误: {e}")
                    time.sleep(1)
            
        except Exception as e:
            print(f"❌ 任务循环错误: {e}")
    
    def _get_current_flight_data(self) -> Optional[FlightData]:
        """获取当前飞行数据"""
        try:
            status = self.waypoint_manager.get_status()
            if not status['position'] or not status['attitude']:
                return None
            
            pos = status['position']
            att = status['attitude']
            
            return FlightData(
                timestamp=time.time(),
                latitude=pos['lat'],
                longitude=pos['lon'],
                altitude=pos['relative_alt'],
                pitch=att['pitch'],
                roll=att['roll'],
                yaw=att['yaw'],
                ground_speed=status['ground_speed'],
                heading=att['yaw']  # 使用偏航角作为航向
            )
            
        except Exception as e:
            print(f"⚠️ 获取飞行数据失败: {e}")
            return None
    
    def _process_detection_frame(self, frame, flight_data):
        """处理检测帧"""
        try:
            # 使用现有的目标检测系统
            detections = self.strike_system.detector.detect(frame)
            
            if detections is not None and len(detections) > 0:
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    
                    if conf < self.config['conf_threshold']:
                        continue
                    
                    # 计算目标中心点
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # 计算GPS坐标
                    target_gps = self.geo_calculator.pixel_to_gps(
                        center_x, center_y, flight_data, frame.shape[1], frame.shape[0]
                    )
                    
                    if target_gps:
                        # 创建目标信息
                        target_info = {
                            'id': f'T{self.target_counter:04d}',
                            'pixel_pos': (center_x, center_y),
                            'gps_pos': target_gps,
                            'confidence': float(conf),
                            'flight_data': flight_data,
                            'timestamp': time.time(),
                            'bbox': (int(x1), int(y1), int(x2), int(y2))
                        }
                        
                        # 添加到处理队列
                        self.detected_targets.put(target_info)
                        self.target_counter += 1
                        
                        # 绘制检测结果
                        self._draw_detection(frame, target_info)
            
            # 绘制飞行信息
            self._draw_flight_info(frame, flight_data)
            
            return frame
            
        except Exception as e:
            print(f"⚠️ 帧处理错误: {e}")
            return frame
    
    def _draw_detection(self, frame, target_info):
        """绘制检测结果"""
        try:
            bbox = target_info['bbox']
            gps_pos = target_info['gps_pos']
            conf = target_info['confidence']
            
            # 绘制边界框
            color = (0, 255, 0) if conf > self.config['min_target_confidence'] else (0, 255, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 绘制目标信息
            text_lines = [
                f"ID: {target_info['id']}",
                f"GPS: {gps_pos['latitude']:.6f}, {gps_pos['longitude']:.6f}",
                f"置信度: {conf:.3f}"
            ]
            
            y_offset = bbox[1] - 10
            for line in text_lines:
                cv2.putText(frame, line, (bbox[0], y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset -= 20
            
        except Exception as e:
            print(f"⚠️ 绘制检测结果错误: {e}")
    
    def _draw_flight_info(self, frame, flight_data):
        """绘制飞行信息"""
        try:
            # 飞行信息文本
            info_lines = [
                f"GPS: {flight_data.latitude:.6f}, {flight_data.longitude:.6f}",
                f"高度: {flight_data.altitude:.1f}m",
                f"姿态: P{flight_data.pitch:.1f}° R{flight_data.roll:.1f}° Y{flight_data.yaw:.1f}°",
                f"速度: {flight_data.ground_speed:.1f}m/s",
                f"目标数: {len(self.processed_targets)}"
            ]
            
            # 绘制背景
            info_height = len(info_lines) * 25 + 10
            cv2.rectangle(frame, (10, 10), (400, info_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, info_height), (255, 255, 255), 2)
            
            # 绘制文本
            y_pos = 30
            for line in info_lines:
                cv2.putText(frame, line, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 25
            
        except Exception as e:
            print(f"⚠️ 绘制飞行信息错误: {e}")
    
    def _should_process_target(self, target_info) -> bool:
        """判断是否应该处理目标"""
        try:
            # 检查自动投水是否启用
            if not self.config['auto_drop_enabled']:
                return False
            
            # 检查置信度
            if target_info['confidence'] < self.config['min_target_confidence']:
                return False
            
            # 检查冷却时间
            current_time = time.time()
            if current_time - self.last_drop_time < self.config['drop_cooldown']:
                return False
            
            # 检查最大目标数
            if len(self.processed_targets) >= self.config['max_targets_per_mission']:
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ 目标处理判断错误: {e}")
            return False
    
    def _process_target_for_drop(self, target_info):
        """处理目标进行投水"""
        try:
            print(f"🎯 处理投水目标: {target_info['id']}")
            
            gps_pos = target_info['gps_pos']
            
            # 添加目标点和计算投水点
            success = self.waypoint_manager.add_target_and_drop_point(
                gps_pos['latitude'],
                gps_pos['longitude'],
                0.0  # 地面高度
            )
            
            if success:
                print(f"✅ 成功添加投水航点: {target_info['id']}")
                
                # 记录处理的目标
                with self.data_lock:
                    self.processed_targets.append(target_info)
                    self.last_drop_time = time.time()
                
                # 保存数据
                self._save_target_data(target_info)
                
            else:
                print(f"❌ 添加投水航点失败: {target_info['id']}")
            
        except Exception as e:
            print(f"❌ 目标投水处理错误: {e}")
    
    def _save_target_data(self, target_info):
        """保存目标数据"""
        try:
            # 准备保存的数据
            save_data = {
                'target_id': target_info['id'],
                'pixel_position': target_info['pixel_pos'],
                'gps_position': target_info['gps_pos'],
                'confidence': target_info['confidence'],
                'timestamp': target_info['timestamp'],
                'flight_data': {
                    'latitude': target_info['flight_data'].latitude,
                    'longitude': target_info['flight_data'].longitude,
                    'altitude': target_info['flight_data'].altitude,
                    'pitch': target_info['flight_data'].pitch,
                    'roll': target_info['flight_data'].roll,
                    'yaw': target_info['flight_data'].yaw,
                    'ground_speed': target_info['flight_data'].ground_speed,
                    'heading': target_info['flight_data'].heading
                }
            }
            
            # 读取现有数据
            try:
                with open(self.config['save_file'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []
            
            # 添加新数据
            data.append(save_data)
            
            # 保存数据
            with open(self.config['save_file'], 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"⚠️ 保存目标数据错误: {e}")
    
    def _save_data(self):
        """保存所有数据"""
        try:
            print("💾 保存数据...")
            
            # 保存处理过的目标
            with self.data_lock:
                for target in self.processed_targets:
                    self._save_target_data(target)
            
            print(f"✅ 已保存 {len(self.processed_targets)} 个目标数据")
            
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
    
    def _print_target_statistics(self):
        """打印目标统计"""
        try:
            with self.data_lock:
                total_detected = self.target_counter
                total_processed = len(self.processed_targets)
                
                print(f"\n📊 目标统计:")
                print(f"   检测总数: {total_detected}")
                print(f"   处理总数: {total_processed}")
                print(f"   处理率: {total_processed/total_detected*100:.1f}%" if total_detected > 0 else "   处理率: 0%")
                print(f"   自动投水: {'启用' if self.config['auto_drop_enabled'] else '禁用'}")
                
                if self.processed_targets:
                    latest = self.processed_targets[-1]
                    print(f"   最新目标: {latest['id']} (置信度: {latest['confidence']:.3f})")
            
        except Exception as e:
            print(f"⚠️ 统计打印错误: {e}")
    
    def _print_mission_status(self):
        """打印任务状态"""
        try:
            status = self.waypoint_manager.get_status()
            
            print(f"\n📡 任务状态:")
            print(f"   MAVLink连接: {'✅' if status['connected'] else '❌'}")
            print(f"   当前航点: {status['current_waypoint']}/{status['total_waypoints']}")
            print(f"   地面速度: {status['ground_speed']:.1f}m/s")
            
            if status['position']:
                pos = status['position']
                print(f"   当前位置: ({pos['lat']:.6f}, {pos['lon']:.6f}, {pos['relative_alt']:.1f}m)")
            
        except Exception as e:
            print(f"⚠️ 状态打印错误: {e}")
    
    def stop_mission(self):
        """停止任务"""
        try:
            print("🛑 停止集成任务...")
            
            self.is_running = False
            
            # 等待线程结束
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=5)
            
            if self.mission_thread and self.mission_thread.is_alive():
                self.mission_thread.join(timeout=5)
            
            # 保存最终数据
            self._save_data()
            
            # 断开连接
            if self.waypoint_manager:
                self.waypoint_manager.disconnect()
            
            print("✅ 任务已停止")
            
        except Exception as e:
            print(f"❌ 停止任务错误: {e}")

def main():
    """主函数"""
    print("🚁 集成投水任务系统")
    print("=" * 60)
    
    # 配置参数
    config = {
        'mavlink_connection': 'udpin:localhost:14550',  # SITL连接
        'video_source': 'test_video.mp4',  # 视频文件或摄像头ID
        'conf_threshold': 0.25,
        'min_target_confidence': 0.6,
        'auto_drop_enabled': True,
        'drop_cooldown': 10.0,  # 10秒冷却时间
        'max_targets_per_mission': 5,
        'save_file': 'integrated_drop_targets.json'
    }
    
    # 创建任务系统
    mission = IntegratedDropMission(config)
    
    try:
        # 初始化系统
        if not mission.initialize():
            print("❌ 系统初始化失败")
            return
        
        # 启动任务
        if not mission.start_mission():
            print("❌ 任务启动失败")
            return
        
        # 等待任务完成
        try:
            while mission.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠️ 收到中断信号")
        
    finally:
        mission.stop_mission()

if __name__ == "__main__":
    main() 