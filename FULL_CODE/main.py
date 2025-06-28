#!/usr/bin/env python3
# main.py
# 主程序 - 整合GPS通信、目标检测、OCR识别和数据记录功能

import cv2
import numpy as np
import time
import os
from datetime import datetime
import argparse
from pathlib import Path

# 导入自定义模块
from modules.gps_receiver import GPSReceiver
from modules.arrow_processor import ArrowProcessor
from modules.data_recorder import DataRecorder, DetectionRecord
from yolo_trt_utils import YOLOTRTDetector

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='箭头检测与GPS定位系统')
    
    # 视频源设置
    parser.add_argument('--video', type=str, default='0',
                        help='视频源路径，使用数字表示摄像头设备ID')
    
    # 模型设置
    parser.add_argument('--model-dir', type=str, 
                        default='/home/lyc/CQU_Ground_ReconnaissanceStrike/weights',
                        help='模型目录')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='检测置信度阈值')
    
    # GPS设置
    parser.add_argument('--gps-port', type=str, default='/dev/ttyACM0',
                        help='GPS串口设备')
    parser.add_argument('--gps-baud', type=int, default=57600,
                        help='GPS串口波特率')
    parser.add_argument('--no-gps', action='store_true',
                        help='禁用GPS功能')
    
    # 数据存储设置
    parser.add_argument('--data-dir', type=str, default='/home/lyc/detection_data',
                        help='数据存储目录')
    
    # 显示设置
    parser.add_argument('--display', action='store_true', default=True,
                        help='启用图形界面显示')
    parser.add_argument('--width', type=int, default=1280,
                        help='显示窗口宽度')
    parser.add_argument('--height', type=int, default=720,
                        help='显示窗口高度')
    
    # 其他设置
    parser.add_argument('--gpu', action='store_true',
                        help='使用GPU进行OCR')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    print("启动箭头检测与GPS定位系统...")
    
    # 获取模型路径
    model_dir = args.model_dir
    pt_model_path = os.path.join(model_dir, "best.pt")
    trt_model_path = os.path.join(model_dir, "best_trt.engine")
    
    # 检查TensorRT引擎
    if os.path.exists(trt_model_path):
        model_path = trt_model_path
        print(f"使用TensorRT引擎: {trt_model_path}")
    else:
        model_path = pt_model_path
        print(f"使用PyTorch模型: {pt_model_path}")

    # 初始化组件
    try:
        # 初始化检测器
        detector = YOLOTRTDetector(
            model_path=model_path, 
            conf_thres=args.conf_thres, 
            use_trt=os.path.exists(trt_model_path)
        )
        
        # 初始化箭头处理器
        processor = ArrowProcessor(gpu=args.gpu)
        
        # 初始化数据记录器
        recorder = DataRecorder(data_dir=args.data_dir)
        
        # 初始化GPS接收器
        gps_connected = False
        if not args.no_gps:
            gps_receiver = GPSReceiver(
                connection_string=args.gps_port, 
                baud_rate=args.gps_baud
            )
            gps_connected = gps_receiver.connect()
            
            if gps_connected:
                gps_receiver.start_receiving()
                print("GPS接收器启动成功")
            else:
                print("GPS接收器启动失败，将在无GPS模式下运行")
        else:
            print("GPS功能已禁用")
            gps_receiver = None
        
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 视频输入设置
    video_path = args.video
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        print(f"使用摄像头: {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"使用视频文件: {video_path}")
    
    if not cap.isOpened():
        raise IOError(f"无法打开视频源: {video_path}")
    
    # 显示设置
    if args.display:
        cv2.namedWindow("Detection System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection System", args.width, args.height)
    
    # 性能统计
    frame_count = 0
    fps_avg = 0
    start_time = time.time()
    detection_count = 0
    
    print("系统就绪! 按'q'键退出, 按's'键手动保存当前检测")

    try:
        while True:
            loop_start = time.time()
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("视频结束")
                break
                
            # 调整图像大小
            scale_percent = 75
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height))

            # YOLO检测
            detections = detector.detect(frame)
            
            # 获取当前GPS数据
            current_gps = gps_receiver.get_latest_gps() if gps_connected else None
            
            # 计算FPS
            frame_count += 1
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps_avg = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time
            
            # 处理检测结果
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det['box'])
                confidence = det.get('confidence', 0.0)
                
                try:
                    # 扩展检测框
                    expand_ratio = 0.1
                    width_det = x2 - x1
                    height_det = y2 - y1
                    expand_w = int(width_det * expand_ratio)
                    expand_h = int(height_det * expand_ratio)
                    
                    x1_exp = max(0, x1 - expand_w)
                    y1_exp = max(0, y1 - expand_h)
                    x2_exp = min(frame.shape[1], x2 + expand_w)
                    y2_exp = min(frame.shape[0], y2 + expand_h)
                    
                    # 裁剪区域
                    crop = frame[y1_exp:y2_exp, x1_exp:x2_exp].copy()
                    if crop.size == 0:
                        continue
                    
                    # 旋转校正
                    rotated = processor.rotate_arrow(crop)
                    
                    # OCR识别
                    ocr_text, ocr_conf = processor.ocr_recognize(rotated)
                    
                    # 创建检测记录
                    detection_id = recorder.generate_detection_id()
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    record = DetectionRecord(
                        detection_id=detection_id,
                        timestamp=time.time(),
                        datetime_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        pixel_x=center_x,
                        pixel_y=center_y,
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        ocr_text=ocr_text,
                        ocr_confidence=ocr_conf,
                        gps_data=current_gps,
                        image_width=frame.shape[1],
                        image_height=frame.shape[0],
                        crop_image_path=""  # 将在保存时设置
                    )
                    
                    # 自动保存检测记录
                    if ocr_text.strip():  # 只有识别到文字时才保存
                        recorder.save_detection(record, rotated)
                        detection_count += 1
                    
                    # 可视化
                    if args.display:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 显示信息
                        info_text = f"{ocr_text} ({ocr_conf:.2f})"
                        cv2.putText(frame, info_text, (x1, y2 + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # GPS状态
                        if current_gps:
                            gps_text = f"GPS: {current_gps.latitude:.6f}, {current_gps.longitude:.6f}"
                            cv2.putText(frame, gps_text, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # 显示预览
                        preview_size = 100
                        preview = cv2.resize(rotated, (preview_size, preview_size))
                        x_offset = 10 + i * (preview_size + 10)
                        y_offset = 120
                        
                        if x_offset + preview_size <= frame.shape[1] and y_offset + preview_size <= frame.shape[0]:
                            frame[y_offset:y_offset + preview_size, x_offset:x_offset + preview_size] = preview
                    
                except Exception as e:
                    print(f"处理检测异常: {e}")
                    continue
            
            # 显示界面
            if args.display:
                # 显示状态信息
                status_texts = [
                    f"FPS: {fps_avg:.1f}",
                    f"检测数: {detection_count}",
                    f"GPS: {'连接' if current_gps else '断开'}",
                    f"模式: {'TensorRT' if detector.using_trt else 'PyTorch'}"
                ]
                
                for i, text in enumerate(status_texts):
                    cv2.putText(frame, text, (10, 30 + i * 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示当前GPS信息
                if current_gps:
                    gps_info = f"位置: {current_gps.latitude:.6f}, {current_gps.longitude:.6f}, {current_gps.altitude:.1f}m"
                    cv2.putText(frame, gps_info, (10, frame.shape[0] - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    sat_info = f"卫星: {current_gps.satellites_visible}, 航向: {current_gps.heading:.1f}°"
                    cv2.putText(frame, sat_info, (10, frame.shape[0] - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 显示结果
                cv2.imshow("Detection System", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 手动保存当前帧
                    save_path = Path(args.data_dir) / "manual_captures"
                    save_path.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = save_path / f"manual_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"手动保存帧: {filename}")

    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
    
    finally:
        # 清理资源
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        
        if gps_connected:
            gps_receiver.stop()
        
        print(f"\n系统统计:")
        print(f"总检测数: {detection_count}")
        print(f"平均FPS: {fps_avg:.2f}")
        print(f"数据保存目录: {recorder.data_dir}")

if __name__ == "__main__":
    main()  