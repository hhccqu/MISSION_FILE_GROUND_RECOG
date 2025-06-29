#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终SITL连接解决方案
包含多种连接策略和回退机制
"""

import time
import socket
import threading
from pymavlink import mavutil

class SITLConnectionManager:
    """SITL连接管理器"""
    
    def __init__(self):
        self.connection = None
        self.connection_string = None
        self.is_connected = False
        
    def test_all_connections(self):
        """测试所有可能的连接方式"""
        print("🚁 SITL连接管理器 - 全面连接测试")
        print("=" * 60)
        
        # 连接策略列表（按优先级排序）
        strategies = [
            self._try_udp_connections,
            self._try_tcp_connections, 
            self._try_alternative_ports,
            self._setup_udp_listener,
        ]
        
        for strategy in strategies:
            if strategy():
                return True
                
        print("❌ 所有连接策略都失败了")
        return False
    
    def _try_udp_connections(self):
        """尝试UDP连接"""
        print("\n📡 策略1: UDP连接测试")
        
        udp_configs = [
            "udp:127.0.0.1:14550",
            "udp:127.0.0.1:14551", 
            "udpout:127.0.0.1:14550",
            "udpin:127.0.0.1:14550",
        ]
        
        for config in udp_configs:
            if self._test_connection(config):
                return True
        return False
    
    def _try_tcp_connections(self):
        """尝试TCP连接"""
        print("\n🔌 策略2: TCP连接测试") 
        
        tcp_configs = [
            "tcp:127.0.0.1:5760",
            "127.0.0.1:5760",
        ]
        
        for config in tcp_configs:
            if self._test_connection(config, timeout=3):
                return True
        return False
    
    def _try_alternative_ports(self):
        """尝试其他端口"""
        print("\n🔍 策略3: 扫描其他端口")
        
        # 扫描常用MAVLink端口
        ports = [5762, 5763, 14552, 14553, 9999]
        
        for port in ports:
            # 先检查端口是否开放
            if self._check_port_open('127.0.0.1', port):
                print(f"   发现开放端口: {port}")
                
                configs = [
                    f"tcp:127.0.0.1:{port}",
                    f"udp:127.0.0.1:{port}",
                ]
                
                for config in configs:
                    if self._test_connection(config, timeout=2):
                        return True
        return False
    
    def _setup_udp_listener(self):
        """设置UDP监听器"""
        print("\n👂 策略4: 设置UDP监听器")
        
        try:
            # 在标准地面站端口监听
            listener_thread = threading.Thread(
                target=self._udp_listener, 
                args=(14550,),
                daemon=True
            )
            listener_thread.start()
            
            print("   UDP监听器已启动，端口14550")
            print("   请在Mission Planner中配置输出到127.0.0.1:14550")
            
            # 等待一段时间看是否收到数据
            time.sleep(5)
            
            if self.is_connected:
                return True
                
        except Exception as e:
            print(f"   监听器设置失败: {e}")
            
        return False
    
    def _test_connection(self, conn_str, timeout=5):
        """测试单个连接"""
        print(f"   测试: {conn_str}")
        
        try:
            connection = mavutil.mavlink_connection(conn_str)
            heartbeat = connection.wait_heartbeat(timeout=timeout)
            
            if heartbeat:
                print(f"   ✅ 连接成功!")
                print(f"      系统ID: {connection.target_system}")
                print(f"      组件ID: {connection.target_component}")
                
                self.connection = connection
                self.connection_string = conn_str
                self.is_connected = True
                return True
            else:
                connection.close()
                
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            
        return False
    
    def _check_port_open(self, host, port):
        """检查端口是否开放"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    def _udp_listener(self, port):
        """UDP监听器"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('127.0.0.1', port))
            sock.settimeout(10)
            
            data, addr = sock.recvfrom(1024)
            print(f"   ✅ 收到UDP数据来自 {addr}")
            
            # 尝试建立MAVLink连接
            conn_str = f"udp:{addr[0]}:{addr[1]}"
            if self._test_connection(conn_str):
                self.is_connected = True
                
            sock.close()
            
        except Exception as e:
            print(f"   UDP监听失败: {e}")
    
    def get_connection_info(self):
        """获取连接信息"""
        if self.is_connected:
            return {
                'connection_string': self.connection_string,
                'target_system': self.connection.target_system,
                'target_component': self.connection.target_component,
                'connection_object': self.connection
            }
        return None
    
    def close(self):
        """关闭连接"""
        if self.connection:
            self.connection.close()
            self.is_connected = False

def main():
    """主函数"""
    manager = SITLConnectionManager()
    
    if manager.test_all_connections():
        info = manager.get_connection_info()
        print(f"\n🎉 连接成功!")
        print(f"   连接字符串: {info['connection_string']}")
        print(f"   目标系统: {info['target_system']}")
        print(f"   目标组件: {info['target_component']}")
        
        print("\n📡 测试消息接收...")
        for i in range(5):
            msg = manager.connection.recv_match(blocking=True, timeout=2)
            if msg:
                print(f"   消息 {i+1}: {msg.get_type()}")
            else:
                print(f"   消息 {i+1}: 超时")
        
        print(f"\n💡 现在可以在您的代码中使用: {info['connection_string']}")
        
    else:
        print("\n❌ 无法建立SITL连接")
        print("\n🔧 手动解决方案:")
        print("   1. 在Mission Planner中，转到配置/调试")
        print("   2. 添加MAVLink输出: 协议=UDP, 地址=127.0.0.1, 端口=14550")
        print("   3. 重新运行此脚本")
        print("   4. 或者尝试使用模拟模式运行您的打击任务系统")
    
    manager.close()

if __name__ == "__main__":
    main() 