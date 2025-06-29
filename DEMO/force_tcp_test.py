#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制TCP连接测试 - 尝试连接到已有的TCP SITL服务
"""

import time
import socket
from pymavlink import mavutil

def test_raw_tcp_data():
    """测试原始TCP数据接收"""
    print("🔌 测试原始TCP数据接收")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(('127.0.0.1', 5760))
        
        print("✅ TCP连接建立")
        print("等待数据...")
        
        # 接收原始数据
        data = sock.recv(1024)
        if data:
            print(f"✅ 收到数据: {len(data)} 字节")
            print(f"   数据预览: {data[:50]}")
            
            # 检查是否是MAVLink数据
            if data[0] == 0xFE or data[0] == 0xFD:  # MAVLink v1/v2 magic bytes
                print("✅ 这是MAVLink数据!")
                if data[0] == 0xFE:
                    print("   MAVLink v1.0")
                else:
                    print("   MAVLink v2.0")
            else:
                print("❌ 不是标准MAVLink数据")
        else:
            print("❌ 未收到数据")
            
        sock.close()
        return True
        
    except Exception as e:
        print(f"❌ 原始TCP测试失败: {e}")
        return False

def test_mavlink_force_connect():
    """强制MAVLink连接测试"""
    print("\n🛩️ 强制MAVLink连接测试")
    
    connection_methods = [
        ('tcp:127.0.0.1:5760', {}),
        ('127.0.0.1:5760', {}),
        ('tcp:127.0.0.1:5760', {'source_system': 1}),
        ('tcp:127.0.0.1:5760', {'source_system': 254}),
        ('tcp:127.0.0.1:5760', {'source_component': 1}),
        ('tcp:127.0.0.1:5760', {'baud': 57600}),
    ]
    
    for conn_str, kwargs in connection_methods:
        print(f"\n测试: {conn_str} {kwargs}")
        
        try:
            connection = mavutil.mavlink_connection(conn_str, **kwargs)
            print("连接对象创建成功")
            
            # 尝试发送心跳包
            print("发送心跳包...")
            connection.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )
            
            # 等待响应
            print("等待心跳响应 (3秒)...")
            heartbeat = connection.wait_heartbeat(timeout=3)
            
            if heartbeat:
                print("✅ 收到心跳包!")
                print(f"   系统ID: {connection.target_system}")
                print(f"   组件ID: {connection.target_component}")
                
                connection.close()
                return conn_str, kwargs
            else:
                print("❌ 未收到心跳包")
                connection.close()
                
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            continue
    
    return None, None

def test_listen_mode():
    """测试监听模式 - 被动接收数据"""
    print("\n👂 测试监听模式")
    
    try:
        # 创建UDP监听端口
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('127.0.0.1', 14550))  # 标准地面站端口
        sock.settimeout(10)
        
        print("UDP监听端口14550已开启，等待数据...")
        
        data, addr = sock.recvfrom(1024)
        print(f"✅ 收到UDP数据来自 {addr}: {len(data)} 字节")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"❌ UDP监听失败: {e}")
        return False

def main():
    """主函数"""
    print("🚁 强制TCP连接测试")
    print("=" * 50)
    
    # 1. 测试原始TCP数据
    if test_raw_tcp_data():
        print("\n原始TCP连接可用，继续MAVLink测试...")
        
        # 2. 强制MAVLink连接
        conn_str, kwargs = test_mavlink_force_connect()
        
        if conn_str:
            print(f"\n🎉 找到可用连接: {conn_str} {kwargs}")
        else:
            print("\n❌ MAVLink连接失败，尝试监听模式...")
            # 3. 尝试监听模式
            test_listen_mode()
    else:
        print("\n❌ 基础TCP连接失败")
        print("\n💡 建议检查:")
        print("   1. Mission Planner SITL是否正在运行")
        print("   2. SITL是否允许多客户端连接")
        print("   3. 是否需要在SITL中配置输出端口")

if __name__ == "__main__":
    main() 