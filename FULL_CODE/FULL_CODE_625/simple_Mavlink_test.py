#!/usr/bin/env python3

import time
import sys
import serial.tools.list_ports
from pymavlink import mavutil
import argparse

class PixhawkConnector:
    def __init__(self, device=None, baud=57600):
        self.device = device
        self.baud = baud
        self.connection = None
        self.system_id = None
        self.component_id = None

    def list_serial_ports(self):
        """列出所有可用的串口设备"""
        ports = list(serial.tools.list_ports.comports())
        print("\n可用串口设备:")
        for port in ports:
            print(f"设备: {port.device}")
            print(f"描述: {port.description}")
            print(f"硬件ID: {port.hwid}\n")
        return ports

    def auto_detect_pixhawk(self):
        """自动检测Pixhawk设备"""
        ports = self.list_serial_ports()
        for port in ports:
            if "ACM" in port.device or "USB" in port.device:
                print(f"发现可能的Pixhawk设备: {port.device}")
                return port.device
        return None

    def connect(self):
        """建立与Pixhawk的连接"""
        if not self.device:
            self.device = self.auto_detect_pixhawk()
            if not self.device:
                raise Exception("未找到Pixhawk设备")

        print(f"\n尝试连接到设备: {self.device} (波特率: {self.baud})")
        
        try:
            # 尝试建立连接
            self.connection = mavutil.mavlink_connection(
                self.device,
                baud=self.baud,
                source_component=0
            )
            
            print("等待心跳包...")
            self.connection.wait_heartbeat()
            print("收到心跳包！")
            
            # 保存系统和组件ID
            self.system_id = self.connection.target_system
            self.component_id = self.connection.target_component
            
            self.print_connection_info()
            return True
            
        except Exception as e:
            print(f"连接失败: {str(e)}")
            return False

    def print_connection_info(self):
        """打印连接信息"""
        print("\n连接信息:")
        print(f"系统ID: {self.system_id}")
        print(f"组件ID: {self.component_id}")
        print(f"飞控类型: {self.connection.mav_type}")
        
        try:
            print(f"自动驾驶仪类型: {self.connection.autopilot_type}")
        except:
            print("自动驾驶仪类型: 未知")
            
        try:
            print(f"系统状态: {self.connection.flightmode}")
        except:
            print("系统状态: 未知")
        
    def request_parameters(self):
        """请求并显示重要参数"""
        print("\n请求飞控参数...")
        try:
            self.connection.param_fetch_all()
            start_time = time.time()
            
            while True:
                msg = self.connection.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
                if msg:
                    print(f"参数: {msg.param_id} = {msg.param_value}")
                
                if time.time() - start_time > 5:  # 5秒超时
                    break
        except Exception as e:
            print(f"参数请求错误: {str(e)}")

    def request_status(self):
        """请求并显示系统状态"""
        print("\n系统状态:")
        try:
            self.connection.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )
            
            # 请求数据流
            self.connection.mav.request_data_stream_send(
                self.system_id,
                self.component_id,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # 10Hz
                1    # 开启
            )
            
            start_time = time.time()
            while time.time() - start_time < 5:  # 显示5秒的数据
                msg = self.connection.recv_match(blocking=True, timeout=1)
                if msg:
                    msg_type = msg.get_type()
                    print(f"收到消息类型: {msg_type}")
                    
                    if msg_type == "HEARTBEAT":
                        try:
                            print(f"系统状态: {self.connection.flightmode}")
                        except:
                            print("系统状态: 未知")
                    elif msg_type == "SYS_STATUS":
                        try:
                            print(f"电池电量: {msg.battery_remaining}%")
                            print(f"电压: {msg.voltage_battery/1000.0}V")
                        except:
                            print("电池信息: 未知")
                    elif msg_type == "GPS_RAW_INT":
                        try:
                            print(f"GPS修复类型: {msg.fix_type}")
                            print(f"卫星数量: {msg.satellites_visible}")
                        except:
                            print("GPS信息: 未知")
                    elif msg_type == "ATTITUDE":
                        try:
                            print(f"姿态 - Roll: {msg.roll:.2f}, Pitch: {msg.pitch:.2f}, Yaw: {msg.yaw:.2f}")
                        except:
                            print("姿态信息: 未知")
        except Exception as e:
            print(f"状态请求错误: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Pixhawk MAVLink连接测试工具')
    parser.add_argument('--device', type=str, help='串口设备路径 (例如: /dev/ttyACM0)')
    parser.add_argument('--baud', type=int, default=57600, help='波特率 (默认: 57600)')
    args = parser.parse_args()

    connector = PixhawkConnector(args.device, args.baud)
    
    try:
        if connector.connect():
            print("\n连接成功！开始测试...")
            connector.request_parameters()
            connector.request_status()
            
            print("\n保持连接并监控数据流 (按Ctrl+C退出)...")
            while True:
                connector.request_status()
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
    finally:
        if connector.connection:
            connector.connection.close()
            print("连接已关闭")

if __name__ == "__main__":
    main()