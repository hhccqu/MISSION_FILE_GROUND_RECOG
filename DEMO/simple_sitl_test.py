#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的SITL连接测试
"""

import socket
import time
from pymavlink import mavutil

def test_socket_connection():
    """测试基础socket连接"""
    print("🔌 测试基础Socket连接到127.0.0.1:5760")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 5760))
        
        if result == 0:
            print("✅ Socket连接成功")
            sock.close()
            return True
        else:
            print(f"❌ Socket连接失败，错误代码: {result}")
            sock.close()
            return False
            
    except Exception as e:
        print(f"❌ Socket连接异常: {e}")
        return False

def test_mavlink_simple():
    """简单的MAVLink连接测试"""
    print("\n🛩️ 测试MAVLink连接")
    
    try:
        # 使用最简单的连接方式
        connection = mavutil.mavlink_connection('127.0.0.1:5760')
        print("MAVLink连接对象创建成功")
        
        # 设置较短的超时时间
        print("等待心跳包 (5秒超时)...")
        heartbeat = connection.wait_heartbeat(timeout=5)
        
        if heartbeat:
            print("✅ 收到心跳包!")
            print(f"   消息类型: {heartbeat.get_type()}")
            print(f"   系统ID: {connection.target_system}")
            print(f"   组件ID: {connection.target_component}")
            
            # 尝试接收几条消息
            print("\n接收消息测试...")
            for i in range(5):
                msg = connection.recv_match(blocking=True, timeout=2)
                if msg:
                    print(f"   消息 {i+1}: {msg.get_type()}")
                else:
                    print(f"   消息 {i+1}: 超时")
            
            connection.close()
            return True
        else:
            print("❌ 未收到心跳包")
            connection.close()
            return False
            
    except Exception as e:
        print(f"❌ MAVLink连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚁 简单SITL连接测试")
    print("=" * 40)
    
    # 先测试基础连接
    if test_socket_connection():
        # 再测试MAVLink连接
        test_mavlink_simple()
    else:
        print("\n💡 建议:")
        print("   1. 确认Mission Planner SITL正在运行")
        print("   2. 检查SITL界面是否显示'Connected'")
        print("   3. 尝试重启SITL")

if __name__ == "__main__":
    main() 