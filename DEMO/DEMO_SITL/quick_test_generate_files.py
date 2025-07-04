#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成JSON文件测试脚本
运行几秒钟就自动生成 raw_detections.json 和 dual_thread_results.json
"""

import sys
import os
import time
import threading

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dual_thread_sitl_mission import DualThreadSITLMission

def main():
    """快速生成文件测试"""
    print("🚀 快速生成JSON文件测试")
    print("=" * 40)
    
    # 配置参数 - 优化为快速运行
    config = {
        'conf_threshold': 0.2,  # 降低阈值以快速检测到目标
        'camera_fov_h': 60.0,
        'camera_fov_v': 45.0,
        'min_confidence': 0.3,
        'max_targets_per_frame': 3,  # 限制目标数量
        'detection_queue_size': 50,
        'result_queue_size': 20,
        'queue_wait_timeout': 3.0,
        'raw_data_file': 'raw_detections.json',
        'final_results_file': 'dual_thread_results.json',
        'median_coordinates_file': 'median_coordinates.json'
    }
    
    # 视频源
    video_source = "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/video2.mp4"
    
    print("📋 快速测试配置:")
    print(f"  视频源: {video_source}")
    print(f"  将生成: {config['raw_data_file']}")
    print(f"  将生成: {config['final_results_file']}")
    print(f"  将生成: {config['median_coordinates_file']}")
    print()
    
    # 创建任务系统
    mission = DualThreadSITLMission(config)
    
    try:
        # 初始化系统
        print("🔄 初始化系统...")
        mission.initialize()
        
        print("⏱️ 开始快速测试（将在10秒后自动保存并退出）...")
        
        # 启动自动停止定时器
        def auto_stop():
            time.sleep(10)  # 10秒后自动停止
            print("\n⏰ 10秒测试时间到，自动保存并退出...")
            mission.running = False
        
        timer_thread = threading.Thread(target=auto_stop, daemon=True)
        timer_thread.start()
        
        # 运行任务
        mission.run_video_mission(video_source)
        
        print("\n✅ 快速测试完成！")
        print("📁 生成的文件:")
        
        # 检查生成的文件
        import os
        for filename in [config['raw_data_file'], config['final_results_file'], config['median_coordinates_file']]:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"  ✅ {filename} ({size} bytes)")
            else:
                print(f"  ❌ {filename} (未生成)")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        # 即使中断也要保存数据
        mission._save_current_data()
    except Exception as e:
        print(f"\n❌ 测试错误: {e}")
        # 出错也要尝试保存数据
        try:
            mission._save_current_data()
        except:
            pass

if __name__ == "__main__":
    main() 