#!/usr/bin/env python3
# modules/gps_receiver.py
# GPS接收模块 - 负责与Pixhawk飞控通信获取GPS数据

import time
import threading
import queue
import numpy as np
from dataclasses import dataclass
from typing import Optional

# GPS数据结构
@dataclass
class GPSData:
    """GPS数据结构"""
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    heading: float
    speed: float
    fix_type: int
    satellites_visible: int

# 尝试导入MAVLink库
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    print("警告: pymavlink未安装，GPS功能将被禁用")
    MAVLINK_AVAILABLE = False

class GPSReceiver:
    """GPS数据接收器 - 与Pixhawk通信"""
    
    def __init__(self, connection_string: str = "/dev/ttyACM0", baud_rate: int = 57600):
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.master = None
        self.latest_gps = None
        self.running = False
        self.gps_queue = queue.Queue(maxsize=10)
        
    def connect(self) -> bool:
        """连接到飞控"""
        if not MAVLINK_AVAILABLE:
            print("MAVLink不可用，GPS功能禁用")
            return False
            
        try:
            print(f"连接飞控: {self.connection_string}:{self.baud_rate}")
            self.master = mavutil.mavlink_connection(
                self.connection_string, 
                baud=self.baud_rate,
                timeout=3
            )
            
            # 等待心跳
            print("等待飞控心跳...")
            self.master.wait_heartbeat(timeout=10)
            print("飞控连接成功!")
            
            # 请求GPS数据流
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                1,  # 1Hz
                1   # 启用
            )
            
            return True
            
        except Exception as e:
            print(f"连接飞控失败: {e}")
            return False
    
    def start_receiving(self):
        """启动GPS数据接收线程"""
        if not self.master:
            return False
            
        self.running = True
        self.gps_thread = threading.Thread(target=self._gps_receiver_loop, daemon=True)
        self.gps_thread.start()
        return True
    
    def _gps_receiver_loop(self):
        """GPS数据接收循环"""
        while self.running:
            try:
                msg = self.master.recv_match(
                    type=['GLOBAL_POSITION_INT', 'GPS_RAW_INT'], 
                    blocking=True, 
                    timeout=1
                )
                
                if msg is None:
                    continue
                    
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    gps_data = GPSData(
                        timestamp=time.time(),
                        latitude=msg.lat / 1e7,
                        longitude=msg.lon / 1e7,
                        altitude=msg.alt / 1000.0,
                        heading=msg.hdg / 100.0 if msg.hdg != 65535 else 0.0,
                        speed=np.sqrt(msg.vx**2 + msg.vy**2) / 100.0,
                        fix_type=3,  # 假设3D定位
                        satellites_visible=0  # 需要从GPS_RAW_INT获取
                    )
                    
                    self.latest_gps = gps_data
                    
                    # 更新队列
                    try:
                        self.gps_queue.put_nowait(gps_data)
                    except queue.Full:
                        try:
                            self.gps_queue.get_nowait()  # 移除旧数据
                            self.gps_queue.put_nowait(gps_data)
                        except queue.Empty:
                            pass
                            
                elif msg.get_type() == 'GPS_RAW_INT':
                    if self.latest_gps:
                        self.latest_gps.satellites_visible = msg.satellites_visible
                        self.latest_gps.fix_type = msg.fix_type
                        
            except Exception as e:
                print(f"GPS接收错误: {e}")
                time.sleep(0.1)
    
    def get_latest_gps(self) -> Optional[GPSData]:
        """获取最新GPS数据"""
        return self.latest_gps
    
    def stop(self):
        """停止GPS接收"""
        self.running = False
        if hasattr(self, 'gps_thread'):
            self.gps_thread.join(timeout=2) 