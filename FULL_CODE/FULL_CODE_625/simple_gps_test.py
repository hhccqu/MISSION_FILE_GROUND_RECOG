#!/usr/bin/env python3
# simple_gps_test.py
# 简化版PIXHAWK GPS测试脚本

import time
import sys
from pymavlink import mavutil

def test_pixhawk_connection(port=None, baudrate=57600):
    """测试PIXHAWK连接和GPS数据"""
    
    print("🚁 简化版PIXHAWK GPS测试")
    print("="*40)
    
    # 如果没有指定端口，尝试常见端口
    if not port:
        test_ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1', '/dev/ttyACM1']
        print("🔍 自动搜索端口...")
    else:
        test_ports = [port]
    
    connection = None
    
    # 尝试连接
    for test_port in test_ports:
        try:
            print(f"   尝试连接: {test_port}")
            connection = mavutil.mavlink_connection(test_port, baud=baudrate)
            
            # 等待心跳包
            print("   等待心跳包...")
            heartbeat = connection.wait_heartbeat(timeout=5)
            
            if heartbeat:
                print(f"✅ 成功连接到: {test_port}")
                print(f"   系统ID: {heartbeat.get_srcSystem()}")
                print(f"   组件ID: {heartbeat.get_srcComponent()}")
                break
            else:
                print(f"   无心跳包")
                connection = None
                
        except Exception as e:
            print(f"   连接失败: {e}")
            connection = None
            continue
    
    if not connection:
        print("❌ 无法连接到PIXHAWK")
        print("\n💡 故障排除:")
        print("   1. 检查PIXHAWK是否连接到Jetson")
        print("   2. 检查USB线缆是否正常")
        print("   3. 确认串口权限: sudo usermod -a -G dialout $USER")
        print("   4. 尝试不同的波特率: 57600, 115200")
        print("   5. 重启PIXHAWK和Jetson")
        return
    
    print(f"\n📡 开始接收GPS数据...")
    print("按Ctrl+C退出")
    print("-"*40)
    
    try:
        message_count = 0
        gps_count = 0
        last_print_time = time.time()
        
        while True:
            # 接收消息
            msg = connection.recv_match(blocking=True, timeout=1)
            
            if msg is None:
                continue
                
            message_count += 1
            msg_type = msg.get_type()
            
            # 每秒打印一次统计
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                print(f"📊 消息统计: {message_count} 条消息, GPS: {gps_count} 条")
                last_print_time = current_time
            
            # 处理GPS相关消息
            if msg_type == 'GLOBAL_POSITION_INT':
                gps_count += 1
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                alt = msg.alt / 1000.0
                relative_alt = msg.relative_alt / 1000.0
                
                print(f"🌍 GPS位置:")
                print(f"   纬度: {lat:.7f}°")
                print(f"   经度: {lon:.7f}°") 
                print(f"   海拔: {alt:.2f}m")
                print(f"   相对高度: {relative_alt:.2f}m")
                print("-"*40)
                
            elif msg_type == 'GPS_RAW_INT':
                fix_types = {0: "无信号", 1: "无定位", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK浮点", 6: "RTK固定"}
                fix_status = fix_types.get(msg.fix_type, f"未知({msg.fix_type})")
                
                print(f"📡 GPS状态:")
                print(f"   定位状态: {fix_status}")
                print(f"   卫星数: {msg.satellites_visible}")
                print(f"   水平精度: {msg.eph/100.0 if msg.eph != 65535 else 0:.2f}")
                print("-"*40)
                
            elif msg_type == 'HEARTBEAT':
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                print(f"💓 心跳包 - 解锁状态: {'✅ 已解锁' if armed else '🔒 已锁定'}")
                
    except KeyboardInterrupt:
        print("\n⏹️  退出程序...")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
    finally:
        if connection:
            connection.close()
        print("👋 程序结束")

if __name__ == "__main__":
    # 解析命令行参数
    port = None
    baudrate = 57600
    
    if len(sys.argv) > 1:
        port = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            baudrate = int(sys.argv[2])
        except ValueError:
            print("警告: 无效的波特率，使用默认值57600")
    
    print(f"使用参数: 端口={port or '自动检测'}, 波特率={baudrate}")
    print("用法: python3 simple_gps_test.py [端口] [波特率]")
    print("示例: python3 simple_gps_test.py /dev/ttyUSB0 115200")
    print()
    
    test_pixhawk_connection(port, baudrate) 