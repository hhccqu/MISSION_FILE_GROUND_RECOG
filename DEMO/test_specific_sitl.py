#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门测试TCP 5760端口的SITL连接
"""

import time
import sys
from pymavlink import mavutil

def test_tcp_5760():
    """测试TCP 5760端口连接"""
    connection_strings = [
        "tcp:localhost:5760",
        "tcp:127.0.0.1:5760",
        "tcp:0.0.0.0:5760",
    ]
    
    print("🛩️ 测试Mission Planner SITL TCP连接")
    print("=" * 50)
    
    for conn_str in connection_strings:
        print(f"\n🔗 测试连接: {conn_str}")
        
        try:
            # 创建连接
            connection = mavutil.mavlink_connection(
                conn_str,
                source_system=255,
                source_component=0
            )
            
            print("⏳ 等待心跳包 (15秒超时)...")
            heartbeat = connection.wait_heartbeat(timeout=15)
            
            if heartbeat:
                print("✅ 连接成功!")
                print(f"   系统ID: {connection.target_system}")
                print(f"   组件ID: {connection.target_component}")
                print(f"   飞控类型: {heartbeat.type}")
                print(f"   自驾仪: {heartbeat.autopilot}")
                print(f"   基础模式: {heartbeat.base_mode}")
                print(f"   自定义模式: {heartbeat.custom_mode}")
                print(f"   系统状态: {heartbeat.system_status}")
                
                # 测试接收消息
                print("\n📡 接收消息测试 (10秒)...")
                start_time = time.time()
                message_count = 0
                message_types = set()
                
                while time.time() - start_time < 10:
                    msg = connection.recv_match(blocking=True, timeout=1)
                    if msg:
                        message_count += 1
                        msg_type = msg.get_type()
                        message_types.add(msg_type)
                        
                        if message_count <= 10:  # 显示前10条
                            print(f"   收到: {msg_type}")
                        
                        # 特别关注GPS和姿态消息
                        if msg_type == 'GLOBAL_POSITION_INT':
                            lat = msg.lat / 1e7
                            lon = msg.lon / 1e7
                            alt = msg.alt / 1000.0
                            print(f"   📍 GPS: {lat:.6f}, {lon:.6f}, 高度: {alt:.1f}m")
                        
                        elif msg_type == 'ATTITUDE':
                            import math
                            roll = math.degrees(msg.roll)
                            pitch = math.degrees(msg.pitch)
                            yaw = math.degrees(msg.yaw)
                            print(f"   🛩️ 姿态: Roll={roll:.1f}° Pitch={pitch:.1f}° Yaw={yaw:.1f}°")
                
                print(f"\n📊 统计:")
                print(f"   总消息数: {message_count}")
                print(f"   消息类型: {len(message_types)}")
                print(f"   消息类型列表: {sorted(message_types)}")
                
                connection.close()
                print(f"\n✅ {conn_str} 连接测试成功！")
                return conn_str
                
            else:
                print("❌ 未收到心跳包")
                connection.close()
                
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            continue
    
    print("\n❌ 所有TCP连接尝试都失败了")
    return None

def main():
    """主函数"""
    successful_connection = test_tcp_5760()
    
    if successful_connection:
        print(f"\n🎉 找到可用连接: {successful_connection}")
        print("\n💡 现在可以使用这个连接字符串运行SITL打击任务系统")
        print(f"   修改 sitl_strike_mission.py 中的连接字符串为: {successful_connection}")
    else:
        print("\n🔧 故障排除建议:")
        print("   1. 确认Mission Planner SITL正在运行")
        print("   2. 检查SITL是否显示'Connected'状态")
        print("   3. 确认TCP端口5760没有被其他程序占用")
        print("   4. 尝试重启Mission Planner SITL")
        print("   5. 检查Windows防火墙设置")

if __name__ == "__main__":
    main() 