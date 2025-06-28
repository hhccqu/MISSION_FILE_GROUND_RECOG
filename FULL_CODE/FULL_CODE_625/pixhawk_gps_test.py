#!/usr/bin/env python3
# pixhawk_gps_test.py
# Jetson Orin Nano 实时接收PIXHAWK GPS和高度信息测试代码

import time
import threading
import json
import argparse
from datetime import datetime
from pymavlink import mavutil
import serial.tools.list_ports

class PixhawkGPSReceiver:
    """PIXHAWK GPS和高度信息接收器"""
    
    def __init__(self, device=None, baudrate=57600):
        """
        初始化PIXHAWK连接
        
        Args:
            device: 设备路径 (如: '/dev/ttyUSB0', '/dev/ttyACM0')
            baudrate: 波特率
        """
        self.device = device
        self.baudrate = baudrate
        self.connection = None
        self.is_connected = False
        self.is_running = False
        self.system_id = None
        self.component_id = None
        
        # GPS数据
        self.gps_data = {
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0,
            'relative_altitude': 0.0,
            'gps_fix_type': 0,
            'satellites_visible': 0,
            'hdop': 0.0,
            'vdop': 0.0,
            'ground_speed': 0.0,
            'course': 0.0,
            'timestamp': None
        }
        
        # 系统状态
        self.system_status = {
            'armed': False,
            'mode': 'UNKNOWN',
            'battery_voltage': 0.0,
            'battery_current': 0.0,
            'battery_remaining': 0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'timestamp': None
        }
        
        # 数据接收线程
        self.receive_thread = None
        self.data_lock = threading.Lock()
        
        # 统计信息
        self.message_count = 0
        self.last_heartbeat = None
        
    def list_serial_ports(self):
        """列出所有可用的串口设备"""
        ports = list(serial.tools.list_ports.comports())
        print("🔍 搜索可用串口设备:")
        for port in ports:
            print(f"   设备: {port.device}")
            print(f"   描述: {port.description}")
            print(f"   硬件ID: {port.hwid}")
            print()
        return ports

    def auto_detect_pixhawk(self):
        """自动检测PIXHAWK设备"""
        print("🔍 自动检测PIXHAWK设备...")
        ports = self.list_serial_ports()
        
        # 优先检查常见的PIXHAWK设备描述
        pixhawk_keywords = ['PX4', 'ArduPilot', 'PIXHAWK', 'FMU']
        
        for port in ports:
            # 检查描述中是否包含PIXHAWK相关关键词
            for keyword in pixhawk_keywords:
                if keyword.lower() in port.description.lower():
                    print(f"✅ 发现PIXHAWK设备: {port.device} ({port.description})")
                    return port.device
        
        # 如果没有找到明确的PIXHAWK设备，尝试ACM和USB设备
        for port in ports:
            if "ACM" in port.device or "USB" in port.device:
                print(f"⚠️  发现可能的PIXHAWK设备: {port.device} ({port.description})")
                return port.device
        
        print("❌ 未找到PIXHAWK设备")
        return None

    def connect(self):
        """连接到PIXHAWK"""
        print("🔗 连接PIXHAWK...")
        
        # 如果没有指定设备，自动检测
        if not self.device:
            self.device = self.auto_detect_pixhawk()
            if not self.device:
                print("❌ 未找到PIXHAWK设备")
                return False
        
        print(f"   尝试连接到设备: {self.device} (波特率: {self.baudrate})")
        
        try:
            # 建立连接
            self.connection = mavutil.mavlink_connection(
                self.device,
                baud=self.baudrate,
                source_component=0
            )
            
            print("   等待心跳包...")
            self.connection.wait_heartbeat(timeout=10)
            print("   ✅ 收到心跳包!")
            
            # 保存系统和组件ID
            self.system_id = self.connection.target_system
            self.component_id = self.connection.target_component
            self.is_connected = True
            
            self.print_connection_info()
            self.request_data_stream()
            
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def print_connection_info(self):
        """打印连接信息"""
        print("\n📋 连接信息:")
        print(f"   系统ID: {self.system_id}")
        print(f"   组件ID: {self.component_id}")
        
        try:
            print(f"   飞控类型: {self.connection.mav_type}")
        except:
            print("   飞控类型: 未知")
            
        try:
            print(f"   自动驾驶仪类型: {self.connection.autopilot_type}")
        except:
            print("   自动驾驶仪类型: 未知")
            
        try:
            print(f"   飞行模式: {self.connection.flightmode}")
        except:
            print("   飞行模式: 未知")
    
    def request_data_stream(self):
        """请求数据流"""
        print("📡 请求数据流...")
        try:
            # 发送心跳包
            self.connection.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )
            
            # 请求所有数据流，频率10Hz
            self.connection.mav.request_data_stream_send(
                self.system_id,
                self.component_id,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # 10Hz
                1    # 开启
            )
            
            print("   ✅ 数据流请求已发送")
            
        except Exception as e:
            print(f"   ⚠️  数据流请求失败: {e}")
    
    def start_receiving(self):
        """开始接收数据"""
        if not self.is_connected:
            print("❌ 未连接到PIXHAWK")
            return False
        
        self.is_running = True
        self.receive_thread = threading.Thread(target=self._receive_loop)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        print("🚀 开始接收数据...")
        return True
    
    def _receive_loop(self):
        """数据接收循环"""
        while self.is_running and self.is_connected:
            try:
                # 接收消息
                msg = self.connection.recv_match(blocking=True, timeout=1)
                if msg is None:
                    continue
                
                self.message_count += 1
                msg_type = msg.get_type()
                
                # 处理不同类型的消息
                if msg_type == 'HEARTBEAT':
                    self._handle_heartbeat(msg)
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self._handle_gps_position(msg)
                elif msg_type == 'GPS_RAW_INT':
                    self._handle_gps_raw(msg)
                elif msg_type == 'SYS_STATUS':
                    self._handle_system_status(msg)
                elif msg_type == 'BATTERY_STATUS':
                    self._handle_battery_status(msg)
                elif msg_type == 'ATTITUDE':
                    self._handle_attitude(msg)
                
            except Exception as e:
                print(f"接收数据时出错: {e}")
                time.sleep(0.1)
    
    def _handle_heartbeat(self, msg):
        """处理心跳包"""
        with self.data_lock:
            self.last_heartbeat = datetime.now()
            self.system_status['armed'] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            
            # 获取飞行模式
            try:
                self.system_status['mode'] = self.connection.flightmode
            except:
                if hasattr(msg, 'custom_mode'):
                    mode_mapping = {
                        0: 'MANUAL',
                        1: 'CIRCLE',
                        2: 'STABILIZE',
                        3: 'TRAINING',
                        4: 'ACRO',
                        5: 'FBWA',
                        6: 'FBWB',
                        7: 'CRUISE',
                        8: 'AUTOTUNE',
                        10: 'AUTO',
                        11: 'RTL',
                        12: 'LOITER',
                        15: 'GUIDED',
                        16: 'INITIALISING',
                        17: 'QSTABILIZE',
                        18: 'QHOVER',
                        19: 'QLOITER',
                        20: 'QLAND',
                        21: 'QRTL'
                    }
                    self.system_status['mode'] = mode_mapping.get(msg.custom_mode, f'UNKNOWN({msg.custom_mode})')
            
            self.system_status['timestamp'] = datetime.now()
    
    def _handle_gps_position(self, msg):
        """处理GPS位置信息"""
        with self.data_lock:
            self.gps_data['latitude'] = msg.lat / 1e7  # 转换为度
            self.gps_data['longitude'] = msg.lon / 1e7  # 转换为度
            self.gps_data['altitude'] = msg.alt / 1000.0  # 转换为米
            self.gps_data['relative_altitude'] = msg.relative_alt / 1000.0  # 转换为米
            self.gps_data['ground_speed'] = msg.vx / 100.0  # 转换为m/s
            self.gps_data['course'] = msg.hdg / 100.0  # 转换为度
            self.gps_data['timestamp'] = datetime.now()
    
    def _handle_gps_raw(self, msg):
        """处理原始GPS数据"""
        with self.data_lock:
            self.gps_data['gps_fix_type'] = msg.fix_type
            self.gps_data['satellites_visible'] = msg.satellites_visible
            self.gps_data['hdop'] = msg.eph / 100.0 if msg.eph != 65535 else 0.0
            self.gps_data['vdop'] = msg.epv / 100.0 if msg.epv != 65535 else 0.0
    
    def _handle_system_status(self, msg):
        """处理系统状态"""
        with self.data_lock:
            self.system_status['battery_voltage'] = msg.voltage_battery / 1000.0  # 转换为V
            self.system_status['battery_current'] = msg.current_battery / 100.0  # 转换为A
            self.system_status['battery_remaining'] = msg.battery_remaining  # 百分比
    
    def _handle_battery_status(self, msg):
        """处理电池状态"""
        with self.data_lock:
            if len(msg.voltages) > 0:
                self.system_status['battery_voltage'] = msg.voltages[0] / 1000.0
            self.system_status['battery_current'] = msg.current_battery / 100.0
            self.system_status['battery_remaining'] = msg.battery_remaining
    
    def _handle_attitude(self, msg):
        """处理姿态信息"""
        with self.data_lock:
            self.system_status['roll'] = msg.roll
            self.system_status['pitch'] = msg.pitch
            self.system_status['yaw'] = msg.yaw
    
    def get_gps_data(self):
        """获取GPS数据"""
        with self.data_lock:
            return self.gps_data.copy()
    
    def get_system_status(self):
        """获取系统状态"""
        with self.data_lock:
            return self.system_status.copy()
    
    def get_connection_status(self):
        """获取连接状态"""
        return {
            'connected': self.is_connected,
            'running': self.is_running,
            'message_count': self.message_count,
            'last_heartbeat': self.last_heartbeat,
            'device': self.device,
            'system_id': self.system_id,
            'component_id': self.component_id
        }
    
    def stop(self):
        """停止接收数据"""
        print("⏹️  停止接收数据...")
        self.is_running = False
        
        if self.receive_thread:
            self.receive_thread.join(timeout=2)
        
        if self.connection:
            self.connection.close()
        
        self.is_connected = False
        print("✅ 已停止")

def format_gps_coordinate(coord):
    """格式化GPS坐标"""
    if coord == 0:
        return "0.000000°"
    
    degrees = int(coord)
    minutes = (coord - degrees) * 60
    return f"{degrees}°{minutes:.4f}'"

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PIXHAWK GPS和高度信息接收测试')
    parser.add_argument('--device', type=str, help='串口设备路径 (例如: /dev/ttyACM0)')
    parser.add_argument('--baud', type=int, default=57600, help='波特率 (默认: 57600)')
    args = parser.parse_args()
    
    print("🚁 PIXHAWK GPS和高度信息接收测试")
    print("=" * 60)
    
    # 创建接收器
    receiver = PixhawkGPSReceiver(args.device, args.baud)
    
    try:
        # 连接到PIXHAWK
        if not receiver.connect():
            print("❌ 无法连接到PIXHAWK，请检查:")
            print("   1. PIXHAWK是否已连接到Jetson")
            print("   2. 串口权限是否正确 (sudo usermod -a -G dialout $USER)")
            print("   3. 波特率是否匹配")
            print("   4. 设备路径是否正确")
            print("\n💡 使用方法:")
            print("   python3 pixhawk_gps_test.py --device /dev/ttyACM0 --baud 57600")
            return
        
        # 开始接收数据
        if not receiver.start_receiving():
            return
        
        print("\n📡 实时数据显示 (按Ctrl+C退出)")
        print("=" * 60)
        
        last_display_time = 0
        display_interval = 1.0  # 每秒更新一次显示
        
        while True:
            current_time = time.time()
            
            # 控制显示频率
            if current_time - last_display_time >= display_interval:
                # 清屏
                print("\033[2J\033[H", end="")
                
                # 获取数据
                gps_data = receiver.get_gps_data()
                system_status = receiver.get_system_status()
                connection_status = receiver.get_connection_status()
                
                # 显示连接状态
                print("🔗 连接状态")
                print("-" * 30)
                print(f"连接: {'✅ 已连接' if connection_status['connected'] else '❌ 未连接'}")
                print(f"设备: {connection_status['device']}")
                print(f"系统ID: {connection_status['system_id']}")
                print(f"组件ID: {connection_status['component_id']}")
                print(f"消息数: {connection_status['message_count']}")
                if connection_status['last_heartbeat']:
                    heartbeat_age = (datetime.now() - connection_status['last_heartbeat']).total_seconds()
                    print(f"心跳: {heartbeat_age:.1f}秒前")
                
                # 显示GPS信息
                print(f"\n📍 GPS信息")
                print("-" * 30)
                
                # GPS状态
                fix_types = {
                    0: "无信号",
                    1: "无定位",
                    2: "2D定位",
                    3: "3D定位",
                    4: "DGPS",
                    5: "RTK浮点",
                    6: "RTK固定"
                }
                
                fix_type = gps_data['gps_fix_type']
                fix_status = fix_types.get(fix_type, f"未知({fix_type})")
                
                print(f"定位状态: {fix_status}")
                print(f"卫星数量: {gps_data['satellites_visible']}")
                print(f"水平精度: {gps_data['hdop']:.2f}")
                print(f"垂直精度: {gps_data['vdop']:.2f}")
                
                # 位置信息
                if gps_data['latitude'] != 0 or gps_data['longitude'] != 0:
                    print(f"\n📍 位置信息:")
                    print(f"纬度: {gps_data['latitude']:.7f}° ({format_gps_coordinate(gps_data['latitude'])})")
                    print(f"经度: {gps_data['longitude']:.7f}° ({format_gps_coordinate(gps_data['longitude'])})")
                    print(f"海拔高度: {gps_data['altitude']:.2f}m")
                    print(f"相对高度: {gps_data['relative_altitude']:.2f}m")
                    print(f"地面速度: {gps_data['ground_speed']:.2f}m/s")
                    print(f"航向角: {gps_data['course']:.1f}°")
                else:
                    print("位置: 无GPS信号")
                
                # 显示系统状态
                print(f"\n🔋 系统状态")
                print("-" * 30)
                print(f"飞行模式: {system_status['mode']}")
                print(f"解锁状态: {'✅ 已解锁' if system_status['armed'] else '🔒 已锁定'}")
                print(f"电池电压: {system_status['battery_voltage']:.2f}V")
                print(f"电池电流: {system_status['battery_current']:.2f}A")
                print(f"电池剩余: {system_status['battery_remaining']}%")
                
                # 显示姿态信息
                if system_status['roll'] != 0 or system_status['pitch'] != 0 or system_status['yaw'] != 0:
                    print(f"\n🎯 姿态信息")
                    print("-" * 30)
                    print(f"横滚角: {system_status['roll']:.2f}°")
                    print(f"俯仰角: {system_status['pitch']:.2f}°")
                    print(f"偏航角: {system_status['yaw']:.2f}°")
                
                # 显示时间戳
                if gps_data['timestamp']:
                    print(f"\n⏰ 最后更新: {gps_data['timestamp'].strftime('%H:%M:%S')}")
                
                print(f"\n按Ctrl+C退出...")
                
                last_display_time = current_time
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  收到退出信号...")
    
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
    
    finally:
        # 停止接收器
        receiver.stop()
        print("👋 程序结束")

if __name__ == "__main__":
    main() 