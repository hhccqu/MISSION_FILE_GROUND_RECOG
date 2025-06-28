#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAVLink打击指挥官
连接Pixhawk飞控，发送打击目标GPS坐标
"""

import time
import threading
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

# 尝试导入MAVLink库
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: pymavlink未安装，MAVLink功能将被禁用")
    print("   安装命令: pip install pymavlink")
    MAVLINK_AVAILABLE = False

@dataclass
class TargetCoordinate:
    """目标坐标信息"""
    latitude: float
    longitude: float
    altitude: float = 0.0
    target_id: str = ""
    confidence: float = 0.0
    timestamp: float = 0.0

@dataclass
class FlightStatus:
    """飞行状态信息"""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    relative_altitude: float = 0.0
    ground_speed: float = 0.0
    heading: float = 0.0
    armed: bool = False
    mode: str = "UNKNOWN"
    gps_fix: int = 0
    satellites: int = 0

class MissionType(Enum):
    """任务类型"""
    WAYPOINT = 1
    LOITER = 2
    RTL = 3
    LAND = 4
    TAKEOFF = 5

class MAVLinkStrikeCommander:
    """MAVLink打击指挥官"""
    
    def __init__(self, connection_string: str = "/dev/ttyACM0", baud_rate: int = 57600, 
                 simulation_mode: bool = False):
        """
        初始化MAVLink连接
        
        参数:
            connection_string: 连接字符串 (串口路径或UDP地址)
            baud_rate: 波特率
            simulation_mode: 是否强制使用模拟模式
        """
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.connection = None
        self.is_connected = False
        self.is_monitoring = False
        self.simulation_mode = simulation_mode  # 新增：模拟模式标志
        
        # 飞行状态
        self.flight_status = FlightStatus()
        self.status_lock = threading.Lock()
        
        # 监控线程
        self.monitor_thread = None
        
        # 统计信息
        self.message_count = 0
        self.last_heartbeat = 0
        
        # 模拟模式相关
        self.sim_start_time = time.time()
        self.sim_flight_data = {
            'start_lat': 30.6586,
            'start_lon': 104.0647,
            'altitude': 100.0,
            'speed': 30.0,
            'heading': 90.0
        }
        
        # 如果强制模拟模式，直接启动模拟
        if self.simulation_mode:
            print("🎮 强制模拟模式已启用")
            self._start_simulation()
    
    def connect(self) -> bool:
        """连接到飞控，失败时自动切换到模拟模式"""
        if self.simulation_mode:
            print("🎮 使用模拟模式，跳过真实飞控连接")
            return True
            
        if not MAVLINK_AVAILABLE:
            print("❌ MAVLink库不可用，切换到模拟模式")
            return self._fallback_to_simulation()
        
        try:
            print(f"🔗 正在连接飞控: {self.connection_string}")
            
            # 创建连接
            if self.connection_string.startswith('udp:'):
                self.connection = mavutil.mavlink_connection(self.connection_string)
            else:
                self.connection = mavutil.mavlink_connection(
                    self.connection_string, 
                    baud=self.baud_rate,
                    timeout=3
                )
            
            # 等待心跳包
            print("⏳ 等待飞控心跳包...")
            heartbeat = self.connection.wait_heartbeat(timeout=10)
            
            if heartbeat:
                print(f"✅ 飞控连接成功!")
                print(f"   系统ID: {self.connection.target_system}")
                print(f"   组件ID: {self.connection.target_component}")
                print(f"   飞控类型: {heartbeat.type}")
                print(f"   自驾仪: {heartbeat.autopilot}")
                
                self.is_connected = True
                
                # 请求数据流
                self._request_data_streams()
                
                # 启动状态监控
                self.start_monitoring()
                
                return True
            else:
                print("❌ 未收到心跳包，连接失败")
                return self._fallback_to_simulation()
                
        except Exception as e:
            print(f"❌ 连接飞控失败: {e}")
            return self._fallback_to_simulation()
    
    def _fallback_to_simulation(self) -> bool:
        """切换到模拟模式"""
        print("🎮 切换到模拟模式")
        print("   - 模拟GPS定位")
        print("   - 模拟飞行状态")
        print("   - 模拟任务执行")
        
        self.simulation_mode = True
        self._start_simulation()
        return True
    
    def _start_simulation(self):
        """启动模拟模式"""
        self.is_connected = True
        self.sim_start_time = time.time()
        
        # 初始化模拟飞行状态
        with self.status_lock:
            self.flight_status.latitude = self.sim_flight_data['start_lat']
            self.flight_status.longitude = self.sim_flight_data['start_lon']
            self.flight_status.altitude = self.sim_flight_data['altitude']
            self.flight_status.relative_altitude = self.sim_flight_data['altitude']
            self.flight_status.ground_speed = self.sim_flight_data['speed']
            self.flight_status.heading = self.sim_flight_data['heading']
            self.flight_status.armed = True
            self.flight_status.mode = "AUTO"
            self.flight_status.gps_fix = 3  # 3D Fix
            self.flight_status.satellites = 12
        
        # 启动模拟监控
        self.start_monitoring()
    
    def _simulate_flight_update(self):
        """模拟飞行状态更新"""
        if not self.simulation_mode:
            return
            
        elapsed = time.time() - self.sim_start_time
        
        # 模拟直线飞行
        distance_traveled = self.sim_flight_data['speed'] * elapsed  # 米
        
        # 将距离转换为经纬度变化（粗略计算）
        lat_change = 0
        lon_change = distance_traveled / 111320.0  # 1度经度约111320米（在赤道附近）
        
        with self.status_lock:
            self.flight_status.longitude = self.sim_flight_data['start_lon'] + lon_change
            # 添加一些随机变化模拟真实飞行
            import random
            self.flight_status.altitude = self.sim_flight_data['altitude'] + random.uniform(-5, 5)
            self.flight_status.ground_speed = self.sim_flight_data['speed'] + random.uniform(-2, 2)
            self.flight_status.heading = self.sim_flight_data['heading'] + random.uniform(-5, 5)
    
    def _request_data_streams(self):
        """请求数据流"""
        if not self.connection:
            return
        
        # 请求位置信息
        self.connection.mav.request_data_stream_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            2,  # 2Hz
            1   # 启用
        )
        
        # 请求系统状态
        self.connection.mav.request_data_stream_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
            2,  # 2Hz
            1   # 启用
        )
    
    def start_monitoring(self):
        """启动状态监控线程"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📡 状态监控已启动")
    
    def _monitor_loop(self):
        """状态监控循环"""
        while self.is_monitoring and self.is_connected:
            try:
                if self.simulation_mode:
                    # 模拟模式：定期更新模拟状态
                    self._simulate_flight_update()
                    self.last_heartbeat = time.time()
                    time.sleep(0.5)  # 2Hz更新频率
                    continue
                
                # 真实飞控模式：接收MAVLink消息
                msg = self.connection.recv_match(blocking=True, timeout=1)
                
                if msg is None:
                    continue
                
                self.message_count += 1
                msg_type = msg.get_type()
                
                # 处理不同类型的消息
                if msg_type == 'HEARTBEAT':
                    self.last_heartbeat = time.time()
                    with self.status_lock:
                        self.flight_status.armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                        
                elif msg_type == 'GLOBAL_POSITION_INT':
                    with self.status_lock:
                        self.flight_status.latitude = msg.lat / 1e7
                        self.flight_status.longitude = msg.lon / 1e7
                        self.flight_status.altitude = msg.alt / 1000.0
                        self.flight_status.relative_altitude = msg.relative_alt / 1000.0
                        self.flight_status.ground_speed = math.sqrt(msg.vx**2 + msg.vy**2) / 100.0
                        self.flight_status.heading = msg.hdg / 100.0 if msg.hdg != 65535 else 0.0
                        
                elif msg_type == 'GPS_RAW_INT':
                    with self.status_lock:
                        self.flight_status.gps_fix = msg.fix_type
                        self.flight_status.satellites = msg.satellites_visible
                        
                elif msg_type == 'SYS_STATUS':
                    # 系统状态处理
                    pass
                    
            except Exception as e:
                print(f"⚠️ 监控线程错误: {e}")
                time.sleep(1)
    
    def get_flight_status(self) -> FlightStatus:
        """获取当前飞行状态"""
        with self.status_lock:
            return FlightStatus(
                latitude=self.flight_status.latitude,
                longitude=self.flight_status.longitude,
                altitude=self.flight_status.altitude,
                relative_altitude=self.flight_status.relative_altitude,
                ground_speed=self.flight_status.ground_speed,
                heading=self.flight_status.heading,
                armed=self.flight_status.armed,
                mode=self.flight_status.mode,
                gps_fix=self.flight_status.gps_fix,
                satellites=self.flight_status.satellites
            )
    
    def send_target_waypoint(self, target: TargetCoordinate, altitude: float = 100.0) -> bool:
        """
        发送目标航点到飞控
        
        参数:
            target: 目标坐标
            altitude: 飞行高度（米）
            
        返回:
            bool: 发送是否成功
        """
        if not self.is_connected:
            print("❌ 飞控未连接")
            return False
        
        print(f"🎯 发送目标航点:")
        print(f"   目标ID: {target.target_id}")
        print(f"   纬度: {target.latitude:.7f}°")
        print(f"   经度: {target.longitude:.7f}°")
        print(f"   高度: {altitude}m")
        
        if self.simulation_mode:
            print("🎮 模拟模式：模拟发送航点到飞控")
            print("   - 航点1: 起飞点")
            print("   - 航点2: 目标点")
            print("   - 航点3: 盘旋等待")
            print("✅ 模拟航点发送成功")
            return True
        
        try:
            # 清除当前任务
            self._clear_mission()
            
            # 发送新的航点任务
            seq = 0
            
            # 航点1: 起飞点
            current_status = self.get_flight_status()
            takeoff_alt = max(altitude, 50.0)  # 最小起飞高度50米
            
            self.connection.mav.mission_item_int_send(
                self.connection.target_system,
                self.connection.target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 1,  # current=0, autocontinue=1
                0, 0, 0, 0,  # param1-4
                int(current_status.latitude * 1e7),
                int(current_status.longitude * 1e7),
                takeoff_alt
            )
            seq += 1
            
            # 航点2: 目标点
            self.connection.mav.mission_item_int_send(
                self.connection.target_system,
                self.connection.target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 1,  # current=0, autocontinue=1
                0, 0, 0, 0,  # param1-4
                int(target.latitude * 1e7),
                int(target.longitude * 1e7),
                altitude
            )
            seq += 1
            
            # 航点3: 盘旋等待
            self.connection.mav.mission_item_int_send(
                self.connection.target_system,
                self.connection.target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
                0, 1,  # current=0, autocontinue=1
                0, 0, 50, 0,  # param1-4 (radius=50m)
                int(target.latitude * 1e7),
                int(target.longitude * 1e7),
                altitude
            )
            
            # 发送任务数量
            self.connection.mav.mission_count_send(
                self.connection.target_system,
                self.connection.target_component,
                seq + 1
            )
            
            print("✅ 目标航点已发送到飞控")
            return True
            
        except Exception as e:
            print(f"❌ 发送航点失败: {e}")
            return False
    
    def _clear_mission(self):
        """清除当前任务"""
        if self.simulation_mode:
            return
            
        try:
            self.connection.mav.mission_clear_all_send(
                self.connection.target_system,
                self.connection.target_component
            )
            time.sleep(0.5)  # 等待清除完成
        except Exception as e:
            print(f"⚠️ 清除任务失败: {e}")
    
    def arm_disarm(self, arm: bool = True) -> bool:
        """解锁/锁定飞控"""
        if not self.is_connected:
            print("❌ 飞控未连接")
            return False
        
        action = "解锁" if arm else "锁定"
        print(f"🔐 {action}飞控...")
        
        if self.simulation_mode:
            print(f"🎮 模拟模式：模拟{action}飞控")
            with self.status_lock:
                self.flight_status.armed = arm
            print(f"✅ 模拟飞控{action}成功")
            return True
        
        try:
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1 if arm else 0,  # param1: 1=arm, 0=disarm
                0, 0, 0, 0, 0, 0  # param2-7
            )
            
            # 等待确认
            time.sleep(2)
            status = self.get_flight_status()
            
            if status.armed == arm:
                print(f"✅ 飞控{action}成功")
                return True
            else:
                print(f"❌ 飞控{action}失败")
                return False
                
        except Exception as e:
            print(f"❌ {action}飞控失败: {e}")
            return False
    
    def set_mode(self, mode: str) -> bool:
        """设置飞行模式"""
        if not self.is_connected:
            print("❌ 飞控未连接")
            return False
        
        print(f"🎮 设置飞行模式: {mode}")
        
        if self.simulation_mode:
            print(f"🎮 模拟模式：模拟设置飞行模式为 {mode}")
            with self.status_lock:
                self.flight_status.mode = mode
            print(f"✅ 模拟飞行模式设置成功")
            return True
        
        try:
            # 获取模式映射
            mode_mapping = self.connection.mode_mapping()
            
            if mode not in mode_mapping:
                print(f"❌ 不支持的飞行模式: {mode}")
                print(f"   支持的模式: {list(mode_mapping.keys())}")
                return False
            
            mode_id = mode_mapping[mode]
            
            self.connection.mav.set_mode_send(
                self.connection.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            
            print(f"✅ 飞行模式设置命令已发送")
            return True
            
        except Exception as e:
            print(f"❌ 设置飞行模式失败: {e}")
            return False
    
    def start_mission(self) -> bool:
        """启动任务"""
        if not self.is_connected:
            print("❌ 飞控未连接")
            return False
        
        print("🚀 启动任务...")
        
        if self.simulation_mode:
            print("🎮 模拟模式：模拟启动任务")
            print("   - 执行起飞")
            print("   - 飞向目标点")
            print("   - 开始盘旋等待")
            print("✅ 模拟任务启动成功")
            return True
        
        try:
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_MISSION_START,
                0,
                0, 0, 0, 0, 0, 0, 0  # param1-7
            )
            
            print("✅ 任务启动命令已发送")
            return True
            
        except Exception as e:
            print(f"❌ 启动任务失败: {e}")
            return False
    
    def emergency_stop(self) -> bool:
        """紧急停止"""
        if not self.is_connected:
            print("❌ 飞控未连接")
            return False
        
        print("🚨 执行紧急停止...")
        
        if self.simulation_mode:
            print("🎮 模拟模式：模拟紧急停止")
            print("   - 停止当前任务")
            print("   - 切换到RTL模式")
            print("   - 返回起飞点")
            self.set_mode("RTL")
            print("✅ 模拟紧急停止成功")
            return True
        
        try:
            # 设置为RTL模式
            self.set_mode("RTL")
            
            print("✅ 紧急停止命令已发送")
            return True
            
        except Exception as e:
            print(f"❌ 紧急停止失败: {e}")
            return False
    
    def print_status(self):
        """打印飞行状态"""
        status = self.get_flight_status()
        
        print(f"\n📊 飞行状态:")
        print(f"   运行模式: {'🎮 模拟模式' if self.simulation_mode else '🔗 真实飞控'}")
        print(f"   连接状态: {'✅ 已连接' if self.is_connected else '❌ 未连接'}")
        print(f"   解锁状态: {'✅ 已解锁' if status.armed else '🔒 已锁定'}")
        print(f"   飞行模式: {status.mode}")
        print(f"   GPS定位: {status.gps_fix} ({status.satellites} 颗卫星)")
        print(f"   当前位置: ({status.latitude:.7f}, {status.longitude:.7f})")
        print(f"   海拔高度: {status.altitude:.1f}m")
        print(f"   相对高度: {status.relative_altitude:.1f}m")
        print(f"   地面速度: {status.ground_speed:.1f}m/s")
        print(f"   航向角: {status.heading:.1f}°")
        
        if self.simulation_mode:
            elapsed = time.time() - self.sim_start_time
            print(f"   模拟时间: {elapsed:.1f}秒")
        else:
            print(f"   消息计数: {self.message_count}")
        
        # 连接健康状态
        if self.last_heartbeat > 0:
            heartbeat_age = time.time() - self.last_heartbeat
            if heartbeat_age < 5:
                print(f"   心跳状态: ✅ 正常 ({heartbeat_age:.1f}s前)")
            else:
                print(f"   心跳状态: ⚠️ 超时 ({heartbeat_age:.1f}s前)")
    
    def disconnect(self):
        """断开连接"""
        print("🔌 断开飞控连接...")
        
        self.is_monitoring = False
        self.is_connected = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        if self.connection:
            self.connection.close()
            self.connection = None
        
        print("✅ 已断开连接")

def main():
    """主函数 - 测试MAVLink连接"""
    print("🎯 MAVLink打击指挥官测试")
    print("=" * 50)
    
    # 创建指挥官
    commander = MAVLinkStrikeCommander()
    
    try:
        # 连接飞控
        if not commander.connect():
            print("❌ 无法连接飞控，退出测试")
            return
        
        # 等待状态稳定
        print("⏳ 等待状态稳定...")
        time.sleep(3)
        
        # 打印状态
        commander.print_status()
        
        # 创建测试目标
        test_target = TargetCoordinate(
            latitude=30.6586,
            longitude=104.0647,
            target_id="TEST_TARGET",
            confidence=0.95,
            timestamp=time.time()
        )
        
        print(f"\n🧪 测试发送目标坐标...")
        success = commander.send_target_waypoint(test_target, altitude=100.0)
        
        if success:
            print("✅ 测试成功!")
        else:
            print("❌ 测试失败!")
        
        # 保持连接一段时间
        print("\n⏳ 保持连接30秒...")
        for i in range(30):
            time.sleep(1)
            if i % 10 == 0:
                commander.print_status()
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"❌ 测试错误: {e}")
    finally:
        commander.disconnect()

if __name__ == "__main__":
    main() 