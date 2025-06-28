#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打击任务执行器
整合目标识别、中位数查找和MAVLink通信
自动找到数字目标的中位数并通过MAVLink发送给飞控
"""

import time
import sys
import os
from typing import Optional, Dict, Any

# 导入自定义模块
from target_median_finder import TargetMedianFinder
from mavlink_strike_commander import MAVLinkStrikeCommander, TargetCoordinate

class StrikeMissionExecutor:
    """打击任务执行器"""
    
    def __init__(self, 
                 data_file: str = "strike_targets.json",
                 mavlink_connection: str = "/dev/ttyACM0",
                 baud_rate: int = 57600,
                 simulation_mode: bool = False):
        """
        初始化打击任务执行器
        
        参数:
            data_file: 目标数据文件路径
            mavlink_connection: MAVLink连接字符串
            baud_rate: 波特率
            simulation_mode: 是否强制使用模拟模式
        """
        self.data_file = data_file
        self.mavlink_connection = mavlink_connection
        self.baud_rate = baud_rate
        self.simulation_mode = simulation_mode
        
        # 组件初始化
        self.median_finder = TargetMedianFinder(data_file)
        self.commander = MAVLinkStrikeCommander(mavlink_connection, baud_rate, simulation_mode)
        
        # 任务状态
        self.median_target = None
        self.mission_active = False
        
    def analyze_targets(self) -> bool:
        """分析目标数据，找到中位数目标"""
        print("🎯 开始分析目标数据...")
        print("=" * 60)
        
        # 加载目标数据
        if not self.median_finder.load_targets_data():
            print("❌ 无法加载目标数据")
            return False
        
        # 提取有效数字目标
        valid_targets = self.median_finder.extract_valid_numbers()
        
        if not valid_targets:
            print("❌ 没有找到有效的数字目标")
            return False
        
        # 显示数字分布
        self.median_finder.print_distribution()
        
        # 找到中位数目标
        self.median_target = self.median_finder.find_median_target()
        
        if not self.median_target:
            print("❌ 未找到中位数目标")
            return False
        
        print(f"\n✅ 中位数目标分析完成!")
        return True
    
    def connect_flight_controller(self) -> bool:
        """连接飞控"""
        print(f"\n🔗 连接飞控...")
        print("=" * 60)
        
        success = self.commander.connect()
        
        if success:
            print("✅ 飞控连接成功!")
            
            # 等待状态稳定
            print("⏳ 等待飞控状态稳定...")
            time.sleep(3)
            
            # 显示飞控状态
            self.commander.print_status()
            
            return True
        else:
            print("❌ 飞控连接失败!")
            return False
    
    def prepare_strike_mission(self, flight_altitude: float = 100.0) -> bool:
        """准备打击任务"""
        if not self.median_target:
            print("❌ 没有中位数目标，无法准备任务")
            return False
        
        if not self.commander.is_connected:
            print("❌ 飞控未连接，无法准备任务")
            return False
        
        print(f"\n🎯 准备打击任务...")
        print("=" * 60)
        
        # 创建目标坐标对象
        target_coord = TargetCoordinate(
            latitude=self.median_target['gps_position']['latitude'],
            longitude=self.median_target['gps_position']['longitude'],
            target_id=self.median_target['target_id'],
            confidence=self.median_target['confidence'],
            timestamp=self.median_target['detection_timestamp']
        )
        
        # 显示目标信息
        print(f"📍 打击目标详细信息:")
        print(f"   目标编号: {target_coord.target_id}")
        print(f"   识别数字: {self.median_target['detected_number']}")
        print(f"   数值: {self.median_target['number']}")
        print(f"   检测置信度: {target_coord.confidence:.3f}")
        print(f"   GPS纬度: {target_coord.latitude:.7f}°")
        print(f"   GPS经度: {target_coord.longitude:.7f}°")
        print(f"   飞行高度: {flight_altitude}m")
        
        # 发送目标航点到飞控
        print(f"\n📡 发送目标航点到飞控...")
        success = self.commander.send_target_waypoint(target_coord, flight_altitude)
        
        if success:
            print("✅ 打击任务准备完成!")
            self.mission_active = True
            return True
        else:
            print("❌ 打击任务准备失败!")
            return False
    
    def execute_mission_interactive(self) -> bool:
        """交互式执行任务"""
        if not self.mission_active:
            print("❌ 任务未准备好")
            return False
        
        print(f"\n🚀 任务执行控制台")
        print("=" * 60)
        print("可用命令:")
        print("  status  - 显示飞控状态")
        print("  arm     - 解锁飞控")
        print("  disarm  - 锁定飞控")
        print("  auto    - 设置自动模式")
        print("  manual  - 设置手动模式")
        print("  start   - 启动任务")
        print("  stop    - 紧急停止")
        print("  quit    - 退出程序")
        print("-" * 60)
        
        while True:
            try:
                command = input("\n🎮 请输入命令: ").strip().lower()
                
                if command == "quit" or command == "q":
                    print("👋 退出任务执行器")
                    break
                
                elif command == "status" or command == "s":
                    self.commander.print_status()
                
                elif command == "arm":
                    self.commander.arm_disarm(True)
                
                elif command == "disarm":
                    self.commander.arm_disarm(False)
                
                elif command == "auto":
                    self.commander.set_mode("AUTO")
                
                elif command == "manual":
                    self.commander.set_mode("MANUAL")
                
                elif command == "start":
                    print("🚀 启动打击任务...")
                    self.commander.start_mission()
                
                elif command == "stop":
                    print("🚨 执行紧急停止...")
                    self.commander.emergency_stop()
                
                elif command == "help" or command == "h":
                    print("📖 帮助信息:")
                    print("   1. 首先检查状态 (status)")
                    print("   2. 解锁飞控 (arm)")
                    print("   3. 设置自动模式 (auto)")
                    print("   4. 启动任务 (start)")
                    print("   5. 如需紧急停止 (stop)")
                
                else:
                    print(f"❓ 未知命令: {command}")
                    print("   输入 'help' 查看帮助")
                
            except KeyboardInterrupt:
                print("\n⚠️ 用户中断")
                break
            except Exception as e:
                print(f"❌ 命令执行错误: {e}")
        
        return True
    
    def auto_execute_mission(self, flight_altitude: float = 100.0, auto_arm: bool = False) -> bool:
        """自动执行完整任务"""
        print(f"\n🤖 自动执行打击任务")
        print("=" * 60)
        
        try:
            # 1. 分析目标
            if not self.analyze_targets():
                return False
            
            # 2. 连接飞控
            if not self.connect_flight_controller():
                return False
            
            # 3. 准备任务
            if not self.prepare_strike_mission(flight_altitude):
                return False
            
            # 4. 显示确认信息
            print(f"\n⚠️ 准备执行打击任务!")
            print(f"   目标: {self.median_target['detected_number']} (ID: {self.median_target['target_id']})")
            print(f"   坐标: ({self.median_target['gps_position']['latitude']:.7f}, {self.median_target['gps_position']['longitude']:.7f})")
            print(f"   高度: {flight_altitude}m")
            
            if not auto_arm:
                confirm = input(f"\n❓ 确认执行任务? (y/N): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    print("❌ 任务已取消")
                    return False
            
            # 5. 执行任务步骤
            print(f"\n🚀 开始执行任务...")
            
            # 检查GPS状态
            status = self.commander.get_flight_status()
            if status.gps_fix < 3:
                print(f"⚠️ GPS定位质量不佳 (fix_type: {status.gps_fix})")
                if not auto_arm:
                    confirm = input("是否继续? (y/N): ").strip().lower()
                    if confirm not in ['y', 'yes']:
                        print("❌ 任务已取消")
                        return False
            
            # 解锁飞控
            if auto_arm or status.armed:
                print("🔓 飞控已解锁或自动解锁")
            else:
                print("🔐 请手动解锁飞控后继续...")
                input("按回车键继续...")
            
            # 设置自动模式
            print("🎮 设置自动模式...")
            self.commander.set_mode("AUTO")
            time.sleep(2)
            
            # 启动任务
            print("🚀 启动打击任务...")
            success = self.commander.start_mission()
            
            if success:
                print("✅ 打击任务已启动!")
                print("📡 任务已发送到飞控，请监控飞行状态")
                
                # 监控任务执行
                print("\n📊 任务监控 (按Ctrl+C停止监控):")
                try:
                    for i in range(60):  # 监控60秒
                        time.sleep(1)
                        if i % 10 == 0:
                            status = self.commander.get_flight_status()
                            print(f"   [{i:2d}s] 位置: ({status.latitude:.6f}, {status.longitude:.6f}) 高度: {status.relative_altitude:.1f}m 速度: {status.ground_speed:.1f}m/s")
                
                except KeyboardInterrupt:
                    print("\n⚠️ 监控已停止")
                
                return True
            else:
                print("❌ 任务启动失败!")
                return False
                
        except Exception as e:
            print(f"❌ 任务执行错误: {e}")
            return False
    
    def cleanup(self):
        """清理资源"""
        print(f"\n🧹 清理资源...")
        
        if self.commander.is_connected:
            self.commander.disconnect()
        
        print("✅ 清理完成")

def main():
    """主函数"""
    print("🎯 无人机打击任务执行器")
    print("=" * 80)
    print("功能: 自动分析目标数据，找到中位数目标，通过MAVLink发送给飞控")
    print("=" * 80)
    
    # 检查数据文件
    data_file = "strike_targets.json"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("   请确保在正确的目录下运行程序")
        return
    
    # 解析命令行参数
    simulation_mode = False
    mavlink_connection = "/dev/ttyACM0"
    
    # 检查是否有模拟模式参数
    if "--sim" in sys.argv or "--simulation" in sys.argv:
        simulation_mode = True
        print("🎮 强制模拟模式已启用")
    
    # Windows系统默认使用COM端口
    if os.name == 'nt':  # Windows
        mavlink_connection = "COM3"  # 根据实际情况修改
    
    # 创建执行器
    executor = StrikeMissionExecutor(
        data_file=data_file,
        mavlink_connection=mavlink_connection,
        baud_rate=57600,
        simulation_mode=simulation_mode
    )
    
    try:
        # 解析命令行参数
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
            
            # 跳过模拟模式参数
            if mode in ["--sim", "--simulation"]:
                mode = sys.argv[2].lower() if len(sys.argv) > 2 else "interactive"
            
            if mode == "auto":
                # 自动模式
                altitude = 100.0
                arg_index = 2
                # 跳过模拟模式参数
                if "--sim" in sys.argv or "--simulation" in sys.argv:
                    arg_index = 3
                
                if len(sys.argv) > arg_index:
                    try:
                        altitude = float(sys.argv[arg_index])
                    except ValueError:
                        print(f"⚠️ 无效的高度参数，使用默认值: {altitude}m")
                
                mode_text = "🎮 模拟" if simulation_mode else "🤖 真实"
                print(f"{mode_text}自动执行模式 (高度: {altitude}m)")
                success = executor.auto_execute_mission(altitude, auto_arm=True)
                
                if success:
                    print("✅ 自动任务执行完成!")
                else:
                    print("❌ 自动任务执行失败!")
            
            elif mode == "analyze":
                # 仅分析模式
                print("📊 仅分析目标数据模式")
                executor.analyze_targets()
            
            elif mode == "test":
                # 测试连接模式
                mode_text = "🎮 模拟" if simulation_mode else "🧪 真实"
                print(f"{mode_text}MAVLink连接测试模式")
                success = executor.connect_flight_controller()
                if success:
                    executor.commander.print_status()
                    if not simulation_mode:
                        input("按回车键断开连接...")
                else:
                    print("💡 提示: 如果没有真实飞控，可以使用模拟模式:")
                    print("   python strike_mission_executor.py --sim test")
            
            else:
                print(f"❓ 未知模式: {mode}")
                print("可用模式:")
                print("   auto [高度]     - 自动执行任务")
                print("   analyze        - 仅分析目标数据")
                print("   test           - 测试MAVLink连接")
                print("   interactive    - 交互模式 (默认)")
                print("参数:")
                print("   --sim          - 强制模拟模式")
                print("示例:")
                print("   python strike_mission_executor.py auto 150")
                print("   python strike_mission_executor.py --sim auto 100")
                print("   python strike_mission_executor.py --sim test")
                return
        
        else:
            # 交互模式 (默认)
            mode_text = "🎮 模拟" if simulation_mode else "🎮 真实"
            print(f"{mode_text}交互模式")
            
            # 分析目标
            if not executor.analyze_targets():
                return
            
            # 连接飞控
            if not executor.connect_flight_controller():
                if not simulation_mode:
                    print("\n💡 提示: 如果没有真实飞控，可以使用模拟模式:")
                    print("   python strike_mission_executor.py --sim")
                return
            
            # 准备任务
            altitude = 100.0
            try:
                alt_input = input(f"\n✈️ 请输入飞行高度 (默认{altitude}m): ").strip()
                if alt_input:
                    altitude = float(alt_input)
            except ValueError:
                print(f"⚠️ 无效输入，使用默认高度: {altitude}m")
            
            if not executor.prepare_strike_mission(altitude):
                return
            
            # 交互式执行
            executor.execute_mission_interactive()
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"❌ 程序错误: {e}")
    finally:
        executor.cleanup()

if __name__ == "__main__":
    main() 