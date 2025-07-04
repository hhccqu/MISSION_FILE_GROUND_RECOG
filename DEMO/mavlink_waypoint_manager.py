#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAVLink航点管理器
功能：
1. 通过MAVLink与飞控通信
2. 获取当前航点任务
3. 插入新的GPS目标点
4. 根据飞行姿态计算投水航点
5. 上传修改后的航点任务
"""

import time
import math
import sys
import threading
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    print("❌ 请安装pymavlink: pip install pymavlink")
    MAVLINK_AVAILABLE = False

@dataclass
class WaypointInfo:
    """航点信息"""
    seq: int                    # 序号
    frame: int                  # 坐标系
    command: int                # 命令类型
    current: int                # 是否为当前航点
    autocontinue: int           # 自动继续
    param1: float               # 参数1
    param2: float               # 参数2
    param3: float               # 参数3
    param4: float               # 参数4（偏航角）
    x: float                    # 纬度/X坐标
    y: float                    # 经度/Y坐标
    z: float                    # 高度/Z坐标
    mission_type: int = 0       # 任务类型

@dataclass
class DropPoint:
    """投水点信息"""
    target_lat: float           # 目标纬度
    target_lon: float           # 目标经度
    drop_lat: float             # 投水纬度
    drop_lon: float             # 投水经度
    drop_altitude: float        # 投水高度
    approach_distance: float    # 接近距离
    wind_compensation: Tuple[float, float] = (0.0, 0.0)  # 风补偿

class MAVLinkWaypointManager:
    """MAVLink航点管理器"""
    
    def __init__(self, connection_string: str = "udpin:localhost:14550"):
        """
        初始化航点管理器
        
        Args:
            connection_string: MAVLink连接字符串
        """
        if not MAVLINK_AVAILABLE:
            raise ImportError("pymavlink库不可用")
            
        self.connection_string = connection_string
        self.connection = None
        self.target_system = None
        self.target_component = None
        self.is_connected = False
        
        # 当前任务信息
        self.current_waypoints = []
        self.mission_count = 0
        self.current_wp_seq = 0
        
        # 飞行状态
        self.current_position = None
        self.current_attitude = None
        self.ground_speed = 0.0
        self.wind_speed = (0.0, 0.0)  # (北向, 东向)
        
        # 投水参数
        self.drop_parameters = {
            'approach_distance': 100.0,      # 接近距离(米)
            'drop_altitude_offset': -10.0,   # 投水高度偏移(米)
            'wind_compensation': True,       # 是否风补偿
            'safety_margin': 20.0,          # 安全边距(米)
        }
        
        self.data_lock = threading.Lock()
        self.monitoring_thread = None
        self.is_monitoring = False
    
    def connect(self) -> bool:
        """连接到飞控"""
        try:
            print(f"🔗 连接飞控: {self.connection_string}")
            
            self.connection = mavutil.mavlink_connection(
                self.connection_string,
                source_system=255,
                source_component=0
            )
            
            print("⏳ 等待心跳包...")
            heartbeat = self.connection.wait_heartbeat(timeout=10)
            
            if heartbeat:
                self.target_system = self.connection.target_system
                self.target_component = self.connection.target_component
                self.is_connected = True
                
                print("✅ 飞控连接成功!")
                print(f"   系统ID: {self.target_system}")
                print(f"   组件ID: {self.target_component}")
                
                # 启动数据监控
                self._start_monitoring()
                
                # 请求当前任务
                self.request_mission_list()
                
                return True
            else:
                print("❌ 未收到心跳包")
                return False
                
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def _start_monitoring(self):
        """启动数据监控"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("📊 数据监控已启动")
    
    def _monitoring_loop(self):
        """数据监控循环"""
        while self.is_monitoring and self.is_connected:
            try:
                msg = self.connection.recv_match(blocking=True, timeout=1)
                if msg is None:
                    continue
                
                msg_type = msg.get_type()
                
                if msg_type == 'GLOBAL_POSITION_INT':
                    with self.data_lock:
                        self.current_position = {
                            'lat': msg.lat / 1e7,
                            'lon': msg.lon / 1e7,
                            'alt': msg.alt / 1000.0,
                            'relative_alt': msg.relative_alt / 1000.0
                        }
                        self.ground_speed = math.sqrt(msg.vx**2 + msg.vy**2) / 100.0
                
                elif msg_type == 'ATTITUDE':
                    with self.data_lock:
                        self.current_attitude = {
                            'roll': math.degrees(msg.roll),
                            'pitch': math.degrees(msg.pitch),
                            'yaw': math.degrees(msg.yaw)
                        }
                
                elif msg_type == 'WIND':
                    with self.data_lock:
                        self.wind_speed = (msg.wind_x, msg.wind_y)
                
                elif msg_type == 'MISSION_CURRENT':
                    with self.data_lock:
                        self.current_wp_seq = msg.seq
                
            except Exception as e:
                print(f"⚠️ 监控错误: {e}")
                time.sleep(1)
    
    def request_mission_list(self) -> bool:
        """请求任务列表"""
        try:
            print("📋 请求当前任务列表...")
            
            self.connection.mav.mission_request_list_send(
                self.target_system,
                self.target_component
            )
            
            # 等待MISSION_COUNT消息
            start_time = time.time()
            while time.time() - start_time < 5:
                msg = self.connection.recv_match(type='MISSION_COUNT', blocking=True, timeout=1)
                if msg:
                    self.mission_count = msg.count
                    print(f"✅ 任务数量: {self.mission_count}")
                    
                    if self.mission_count > 0:
                        return self._download_mission()
                    else:
                        print("⚠️ 当前无任务")
                        return True
            
            print("❌ 请求任务列表超时")
            return False
            
        except Exception as e:
            print(f"❌ 请求任务列表失败: {e}")
            return False
    
    def _download_mission(self) -> bool:
        """下载完整任务"""
        try:
            print("📥 下载任务...")
            self.current_waypoints = []
            
            for seq in range(self.mission_count):
                # 请求特定航点
                self.connection.mav.mission_request_int_send(
                    self.target_system,
                    self.target_component,
                    seq
                )
                
                # 等待MISSION_ITEM_INT消息
                start_time = time.time()
                while time.time() - start_time < 3:
                    msg = self.connection.recv_match(type='MISSION_ITEM_INT', blocking=True, timeout=1)
                    if msg and msg.seq == seq:
                        waypoint = WaypointInfo(
                            seq=msg.seq,
                            frame=msg.frame,
                            command=msg.command,
                            current=msg.current,
                            autocontinue=msg.autocontinue,
                            param1=msg.param1,
                            param2=msg.param2,
                            param3=msg.param3,
                            param4=msg.param4,
                            x=msg.x / 1e7,  # 纬度
                            y=msg.y / 1e7,  # 经度
                            z=msg.z,        # 高度
                            mission_type=msg.mission_type
                        )
                        self.current_waypoints.append(waypoint)
                        break
                else:
                    print(f"❌ 下载航点 {seq} 超时")
                    return False
            
            print(f"✅ 成功下载 {len(self.current_waypoints)} 个航点")
            self._print_waypoints()
            return True
            
        except Exception as e:
            print(f"❌ 下载任务失败: {e}")
            return False
    
    def _print_waypoints(self):
        """打印航点信息"""
        print("\n📍 当前航点列表:")
        for wp in self.current_waypoints:
            cmd_name = self._get_command_name(wp.command)
            if wp.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
                print(f"   {wp.seq:2d}: {cmd_name} -> ({wp.x:.6f}, {wp.y:.6f}, {wp.z:.1f}m)")
            else:
                print(f"   {wp.seq:2d}: {cmd_name}")
    
    def _get_command_name(self, command: int) -> str:
        """获取命令名称"""
        command_names = {
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT: "航点",
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF: "起飞",
            mavutil.mavlink.MAV_CMD_NAV_LAND: "降落",
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH: "返航",
            mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM: "无限盘旋",
            mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME: "定时盘旋",
        }
        return command_names.get(command, f"命令_{command}")
    
    def calculate_drop_point(self, target_lat: float, target_lon: float, 
                           target_alt: float = None) -> DropPoint:
        """
        根据目标位置和当前飞行状态计算投水点
        
        Args:
            target_lat: 目标纬度
            target_lon: 目标经度
            target_alt: 目标高度(可选)
            
        Returns:
            DropPoint: 投水点信息
        """
        if not self.current_position or not self.current_attitude:
            raise ValueError("缺少当前位置或姿态信息")
        
        current_lat = self.current_position['lat']
        current_lon = self.current_position['lon']
        current_alt = self.current_position['relative_alt']
        
        if target_alt is None:
            target_alt = 0.0  # 地面高度
        
        # 计算投水高度
        drop_altitude = current_alt + self.drop_parameters['drop_altitude_offset']
        drop_altitude = max(drop_altitude, target_alt + 50.0)  # 最小安全高度
        
        # 计算飞行方向（从当前位置到目标的方向）
        bearing = self._calculate_bearing(current_lat, current_lon, target_lat, target_lon)
        distance_to_target = self._calculate_distance(current_lat, current_lon, target_lat, target_lon)
        
        # 计算接近距离（基于高度和速度）
        approach_distance = self.drop_parameters['approach_distance']
        
        # 根据高度调整接近距离
        height_factor = (drop_altitude - target_alt) / 100.0  # 每100米增加系数
        approach_distance *= (1.0 + height_factor * 0.2)
        
        # 根据速度调整接近距离
        speed_factor = self.ground_speed / 15.0  # 基准速度15m/s
        approach_distance *= max(0.5, speed_factor)
        
        # 计算投水点位置（在目标点前方）
        drop_lat, drop_lon = self._calculate_point_at_distance(
            target_lat, target_lon, bearing + 180, approach_distance
        )
        
        # 风补偿计算
        wind_compensation = (0.0, 0.0)
        if self.drop_parameters['wind_compensation'] and self.wind_speed != (0.0, 0.0):
            wind_compensation = self._calculate_wind_compensation(
                drop_altitude - target_alt, self.wind_speed
            )
            
            # 应用风补偿
            wind_lat, wind_lon = self._offset_by_meters(drop_lat, drop_lon, 
                                                       wind_compensation[0], wind_compensation[1])
            drop_lat, drop_lon = wind_lat, wind_lon
        
        drop_point = DropPoint(
            target_lat=target_lat,
            target_lon=target_lon,
            drop_lat=drop_lat,
            drop_lon=drop_lon,
            drop_altitude=drop_altitude,
            approach_distance=approach_distance,
            wind_compensation=wind_compensation
        )
        
        print(f"🎯 计算投水点:")
        print(f"   目标位置: ({target_lat:.6f}, {target_lon:.6f})")
        print(f"   投水位置: ({drop_lat:.6f}, {drop_lon:.6f})")
        print(f"   投水高度: {drop_altitude:.1f}m")
        print(f"   接近距离: {approach_distance:.1f}m")
        print(f"   风补偿: 北{wind_compensation[0]:.1f}m, 东{wind_compensation[1]:.1f}m")
        
        return drop_point
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两点间方位角（度）"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两点间距离（米）"""
        R = 6371000  # 地球半径（米）
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat_rad = math.radians(lat2 - lat1)
        dlon_rad = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat_rad/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon_rad/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _calculate_point_at_distance(self, lat: float, lon: float, 
                                   bearing: float, distance: float) -> Tuple[float, float]:
        """计算在指定方向和距离的点"""
        R = 6371000  # 地球半径（米）
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        lat2_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance/R) +
            math.cos(lat_rad) * math.sin(distance/R) * math.cos(bearing_rad)
        )
        
        lon2_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance/R) * math.cos(lat_rad),
            math.cos(distance/R) - math.sin(lat_rad) * math.sin(lat2_rad)
        )
        
        return math.degrees(lat2_rad), math.degrees(lon2_rad)
    
    def _calculate_wind_compensation(self, drop_height: float, wind_speed: Tuple[float, float]) -> Tuple[float, float]:
        """计算风补偿"""
        # 假设投水物下降时间（简化计算）
        fall_time = math.sqrt(2 * drop_height / 9.81)  # 自由落体时间
        
        # 风漂移距离
        wind_drift_north = wind_speed[0] * fall_time
        wind_drift_east = wind_speed[1] * fall_time
        
        # 返回补偿量（与风向相反）
        return (-wind_drift_north, -wind_drift_east)
    
    def _offset_by_meters(self, lat: float, lon: float, 
                         north_meters: float, east_meters: float) -> Tuple[float, float]:
        """根据米偏移计算新坐标"""
        # 地球半径
        R = 6371000
        
        # 纬度偏移
        dlat = north_meters / R
        new_lat = lat + math.degrees(dlat)
        
        # 经度偏移（考虑纬度影响）
        dlon = east_meters / (R * math.cos(math.radians(lat)))
        new_lon = lon + math.degrees(dlon)
        
        return new_lat, new_lon
    
    def insert_drop_waypoint(self, drop_point: DropPoint, insert_after_current: bool = True) -> bool:
        """
        插入投水航点到任务中
        
        Args:
            drop_point: 投水点信息
            insert_after_current: 是否在当前航点后插入
            
        Returns:
            bool: 是否成功
        """
        try:
            if not self.current_waypoints:
                print("❌ 没有当前任务")
                return False
            
            # 确定插入位置
            if insert_after_current:
                insert_seq = self.current_wp_seq + 1
            else:
                insert_seq = len(self.current_waypoints)
            
            # 创建接近航点
            approach_wp = WaypointInfo(
                seq=insert_seq,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                current=0,
                autocontinue=1,
                param1=0,  # 停留时间
                param2=0,  # 接受半径
                param3=0,  # 穿越半径
                param4=0,  # 偏航角
                x=drop_point.drop_lat,
                y=drop_point.drop_lon,
                z=drop_point.drop_altitude
            )
            
            # 创建投水命令航点（可以是DO_SET_SERVO或自定义命令）
            drop_cmd_wp = WaypointInfo(
                seq=insert_seq + 1,
                frame=mavutil.mavlink.MAV_FRAME_MISSION,
                command=mavutil.mavlink.MAV_CMD_DO_SET_SERVO,  # 舵机命令，用于投水
                current=0,
                autocontinue=1,
                param1=9,     # 舵机通道
                param2=1500,  # PWM值（投水）
                param3=0,
                param4=0,
                x=0,
                y=0,
                z=0
            )
            
            # 插入航点
            new_waypoints = []
            seq_counter = 0
            
            for i, wp in enumerate(self.current_waypoints):
                if i == insert_seq:
                    # 插入新航点
                    approach_wp.seq = seq_counter
                    new_waypoints.append(approach_wp)
                    seq_counter += 1
                    
                    drop_cmd_wp.seq = seq_counter
                    new_waypoints.append(drop_cmd_wp)
                    seq_counter += 1
                
                # 添加原有航点
                wp.seq = seq_counter
                new_waypoints.append(wp)
                seq_counter += 1
            
            # 如果在末尾插入
            if insert_seq >= len(self.current_waypoints):
                approach_wp.seq = seq_counter
                new_waypoints.append(approach_wp)
                seq_counter += 1
                
                drop_cmd_wp.seq = seq_counter
                new_waypoints.append(drop_cmd_wp)
                seq_counter += 1
            
            self.current_waypoints = new_waypoints
            
            print(f"✅ 已插入投水航点，总航点数: {len(self.current_waypoints)}")
            self._print_waypoints()
            
            return True
            
        except Exception as e:
            print(f"❌ 插入航点失败: {e}")
            return False
    
    def upload_mission(self) -> bool:
        """上传修改后的任务到飞控"""
        try:
            print("📤 上传任务到飞控...")
            
            # 发送任务数量
            self.connection.mav.mission_count_send(
                self.target_system,
                self.target_component,
                len(self.current_waypoints)
            )
            
            # 等待任务请求
            uploaded_count = 0
            start_time = time.time()
            
            while uploaded_count < len(self.current_waypoints) and time.time() - start_time < 30:
                msg = self.connection.recv_match(type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], 
                                                blocking=True, timeout=2)
                if msg:
                    seq = msg.seq
                    if seq < len(self.current_waypoints):
                        wp = self.current_waypoints[seq]
                        
                        # 发送航点
                        self.connection.mav.mission_item_int_send(
                            self.target_system,
                            self.target_component,
                            wp.seq,
                            wp.frame,
                            wp.command,
                            wp.current,
                            wp.autocontinue,
                            wp.param1,
                            wp.param2,
                            wp.param3,
                            wp.param4,
                            int(wp.x * 1e7),  # 纬度
                            int(wp.y * 1e7),  # 经度
                            wp.z,
                            wp.mission_type
                        )
                        
                        uploaded_count += 1
                        print(f"   上传航点 {seq + 1}/{len(self.current_waypoints)}")
            
            # 等待确认
            msg = self.connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print("✅ 任务上传成功!")
                return True
            else:
                print("❌ 任务上传失败或未确认")
                return False
                
        except Exception as e:
            print(f"❌ 上传任务失败: {e}")
            return False
    
    def add_target_and_drop_point(self, target_lat: float, target_lon: float, 
                                 target_alt: float = None) -> bool:
        """
        添加目标点和计算投水点的完整流程
        
        Args:
            target_lat: 目标纬度
            target_lon: 目标经度
            target_alt: 目标高度
            
        Returns:
            bool: 是否成功
        """
        try:
            print(f"🎯 添加新目标点: ({target_lat:.6f}, {target_lon:.6f})")
            
            # 等待获取当前位置和姿态
            retry_count = 0
            while (not self.current_position or not self.current_attitude) and retry_count < 10:
                print("⏳ 等待飞行数据...")
                time.sleep(1)
                retry_count += 1
            
            if not self.current_position or not self.current_attitude:
                print("❌ 无法获取当前飞行数据")
                return False
            
            # 计算投水点
            drop_point = self.calculate_drop_point(target_lat, target_lon, target_alt)
            
            # 插入航点
            if self.insert_drop_waypoint(drop_point):
                # 上传到飞控
                return self.upload_mission()
            
            return False
            
        except Exception as e:
            print(f"❌ 添加目标点失败: {e}")
            return False
    
    def get_status(self) -> dict:
        """获取当前状态"""
        with self.data_lock:
            return {
                'connected': self.is_connected,
                'position': self.current_position.copy() if self.current_position else None,
                'attitude': self.current_attitude.copy() if self.current_attitude else None,
                'ground_speed': self.ground_speed,
                'wind_speed': self.wind_speed,
                'current_waypoint': self.current_wp_seq,
                'total_waypoints': len(self.current_waypoints)
            }
    
    def disconnect(self):
        """断开连接"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        
        if self.connection:
            self.connection.close()
            self.is_connected = False
        
        print("🔌 已断开连接")

def main():
    """主函数 - 演示用法"""
    print("🚁 MAVLink航点管理器")
    print("=" * 50)
    
    # 创建管理器
    manager = MAVLinkWaypointManager("udpin:localhost:14550")
    
    try:
        # 连接飞控
        if not manager.connect():
            print("❌ 连接失败")
            return
        
        # 等待数据稳定
        print("⏳ 等待飞行数据稳定...")
        time.sleep(3)
        
        # 显示当前状态
        status = manager.get_status()
        print(f"\n📊 当前状态:")
        if status['position']:
            pos = status['position']
            print(f"   位置: ({pos['lat']:.6f}, {pos['lon']:.6f}, {pos['alt']:.1f}m)")
        if status['attitude']:
            att = status['attitude']
            print(f"   姿态: Roll={att['roll']:.1f}° Pitch={att['pitch']:.1f}° Yaw={att['yaw']:.1f}°")
        print(f"   地速: {status['ground_speed']:.1f}m/s")
        print(f"   当前航点: {status['current_waypoint']}/{status['total_waypoints']}")
        
        # 交互模式
        print(f"\n💡 交互命令:")
        print("   add <lat> <lon> [alt] - 添加目标点")
        print("   status - 显示状态")
        print("   mission - 显示任务")
        print("   quit - 退出")
        
        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue
                
                if cmd[0] == 'quit':
                    break
                elif cmd[0] == 'status':
                    status = manager.get_status()
                    print(f"连接状态: {'✅' if status['connected'] else '❌'}")
                    if status['position']:
                        pos = status['position']
                        print(f"位置: ({pos['lat']:.6f}, {pos['lon']:.6f}, {pos['relative_alt']:.1f}m)")
                    print(f"当前航点: {status['current_waypoint']}/{status['total_waypoints']}")
                
                elif cmd[0] == 'mission':
                    manager._print_waypoints()
                
                elif cmd[0] == 'add' and len(cmd) >= 3:
                    lat = float(cmd[1])
                    lon = float(cmd[2])
                    alt = float(cmd[3]) if len(cmd) > 3 else None
                    
                    print(f"🎯 添加目标: ({lat:.6f}, {lon:.6f})")
                    if manager.add_target_and_drop_point(lat, lon, alt):
                        print("✅ 目标点添加成功!")
                    else:
                        print("❌ 目标点添加失败!")
                
                else:
                    print("❌ 无效命令")
                    
            except KeyboardInterrupt:
                break
            except ValueError:
                print("❌ 参数格式错误")
            except Exception as e:
                print(f"❌ 命令执行错误: {e}")
    
    finally:
        manager.disconnect()

if __name__ == "__main__":
    main() 