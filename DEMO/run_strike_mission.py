#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机对地打击任务 - 快速启动脚本
"""

from strike_mission_system import StrikeMissionSystem

def main():
    """快速启动任务"""
    print("🚁 无人机对地打击任务系统")
    print("=" * 50)
    
    # 任务配置
    config = {
        # 检测参数
        'conf_threshold': 0.25,        # YOLO置信度阈值
        'min_confidence': 0.5,         # 目标最小置信度
        'max_targets_per_frame': 5,    # 每帧最大处理目标数
        
        # 相机参数
        'camera_fov_h': 60.0,          # 水平视场角（度）
        'camera_fov_v': 45.0,          # 垂直视场角（度）
        
        # 飞行参数（模拟）
        'start_lat': 30.6586,          # 起始纬度（成都）
        'start_lon': 104.0647,         # 起始经度
        'altitude': 500.0,             # 飞行高度（米）
        'speed': 30.0,                 # 飞行速度（m/s）
        'heading': 90.0,               # 航向角（度，90=东）
        
        # 处理参数
        'ocr_interval': 5,             # OCR处理间隔（帧）
        'save_file': 'strike_targets.json',  # 保存文件
    }
    
    # 视频源（固定使用video2.mp4 - 绝对路径）
    video_sources = [
        "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/video2.mp4",  # 绝对路径
    ]
    
    print("📋 任务配置:")
    print(f"  起始位置: {config['start_lat']:.4f}, {config['start_lon']:.4f}")
    print(f"  飞行高度: {config['altitude']}m")
    print(f"  飞行速度: {config['speed']}m/s")
    print(f"  航向角: {config['heading']}°")
    print(f"  置信度阈值: {config['conf_threshold']}")
    print()
    
    # 创建并初始化任务系统
    mission = StrikeMissionSystem(config)
    
    try:
        print("🔧 正在初始化系统...")
        mission.initialize()
        
        print("\n🎯 开始执行任务...")
        print("按键说明:")
        print("  'q' - 退出任务")
        print("  's' - 保存数据")
        print("  'r' - 重置统计")
        print("  'c' - 清空目标数据")
        print()
        
        # 尝试打开视频源
        for i, source in enumerate(video_sources):
            try:
                print(f"📹 尝试视频源 {i+1}: {source}")
                mission.run_video_mission(source)
                break
            except Exception as e:
                print(f"❌ 视频源失败: {e}")
                if i < len(video_sources) - 1:
                    print("🔄 尝试下一个视频源...")
                continue
        else:
            print("❌ 所有视频源都无法打开")
            return
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断任务")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 任务结束")

if __name__ == "__main__":
    main() 