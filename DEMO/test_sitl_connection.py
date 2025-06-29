#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SITL连接测试工具
用于测试Mission Planner SITL连接
"""

import time
import sys
from pymavlink import mavutil

def test_connection(connection_string, timeout=10):
    """测试单个连接"""
    print(f"🔗 测试连接: {connection_string}")
    
    try:
        # 创建连接
        connection = mavutil.mavlink_connection(
            connection_string,
            source_system=255,
            source_component=0
        )
        
        print("⏳ 等待心跳包...")
        heartbeat = connection.wait_heartbeat(timeout=timeout)
        
        if heartbeat:
            print("✅ 连接成功!")
            print(f"   系统ID: {connection.target_system}")
            print(f"   组件ID: {connection.target_component}")
            print(f"   飞控类型: {heartbeat.type}")
            print(f"   自驾仪: {heartbeat.autopilot}")
            print(f"   基础模式: {heartbeat.base_mode}")
            print(f"   自定义模式: {heartbeat.custom_mode}")
            
            # 测试接收几条消息
            print("\n📡 接收消息测试 (5秒)...")
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < 5:
                msg = connection.recv_match(blocking=True, timeout=1)
                if msg:
                    message_count += 1
                    if message_count <= 5:  # 只显示前5条
                        print(f"   收到: {msg.get_type()}")
            
            print(f"   总计收到 {message_count} 条消息")
            connection.close()
            return True
            
        else:
            print("❌ 未收到心跳包")
            connection.close()
            return False
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def main():
    """主函数"""
    print("🛩️ SITL连接测试工具")
    print("=" * 40)
    
    # 测试不同的连接字符串
    test_connections = [
        # UDP连接（最常用）
        "udpin:localhost:14550",
        "udpin:127.0.0.1:14550", 
        "udp:localhost:14550",
        "udp:127.0.0.1:14550",
        
        # TCP连接
        "tcp:localhost:5760",
        "tcp:127.0.0.1:5760",
        
        # 其他常见端口
        "udpin:localhost:14540",
        "udpin:localhost:14560",
        "tcp:localhost:5761",
        
        # 串口连接（如果使用虚拟串口）
        "COM3:57600",
        "COM4:57600",
        "COM5:57600",
    ]
    
    successful_connections = []
    
    for conn_str in test_connections:
        print(f"\n{'-' * 40}")
        if test_connection(conn_str, timeout=5):
            successful_connections.append(conn_str)
        time.sleep(1)  # 短暂延迟
    
    print(f"\n{'=' * 40}")
    print("📊 测试结果:")
    
    if successful_connections:
        print("✅ 成功的连接:")
        for conn in successful_connections:
            print(f"   {conn}")
        
        print(f"\n💡 建议使用: {successful_connections[0]}")
        
    else:
        print("❌ 没有找到可用的连接")
        print("\n🔧 故障排除:")
        print("   1. 确认Mission Planner SITL正在运行")
        print("   2. 检查SITL输出端口配置")
        print("   3. 确认防火墙允许本地连接")
        print("   4. 尝试重启Mission Planner")
        print("   5. 检查是否有其他程序占用端口")
        
        print("\n📋 Mission Planner SITL配置步骤:")
        print("   1. 打开Mission Planner")
        print("   2. 点击 'Simulation' 选项卡")
        print("   3. 选择飞机类型 (如: ArduPlane)")
        print("   4. 点击 'Multirotor' 或对应类型")
        print("   5. 等待SITL启动完成")
        print("   6. 检查右下角连接状态")

if __name__ == "__main__":
    main() 