#!/usr/bin/env python3
# inference4_simple_realtime.py
# 简化的实时处理版本 - 重点解决跳帧问题

import cv2
import numpy as np
import time
import os
from yolo_trt_utils import YOLOTRTDetector
import easyocr

class ArrowProcessor:
    def __init__(self):
        print("EasyOCR使用CPU模式")
        self.reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        
        # 红色阈值范围（HSV颜色空间）
        self.lower_red1 = np.array([0, 30, 30])
        self.lower_red2 = np.array([150, 30, 30])
        self.upper_red1 = np.array([30, 255, 255])
        self.upper_red2 = np.array([179, 255, 255])
        
        # 形态学处理核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _preprocess_red_mask(self, image):
        """红色区域预处理管道"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        combined = cv2.bitwise_or(mask1, mask2)
        
        # 形态学处理
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return cleaned

    def _correct_rotation(self, image, angle):
        """执行旋转并验证方向"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
        
        # 方向验证（基于红色区域）
        rotated_hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
        rotated_mask1 = cv2.inRange(rotated_hsv, self.lower_red1, self.upper_red1)
        rotated_mask2 = cv2.inRange(rotated_hsv, self.lower_red2, self.upper_red2)
        rotated_mask = cv2.bitwise_or(rotated_mask1, rotated_mask2)
        
        # 比较上下半区
        top = rotated_mask[:h//2, :]
        bottom = rotated_mask[h//2:, :]
        if cv2.countNonZero(bottom) > cv2.countNonZero(top):
            rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        return rotated

    def rotate_arrow(self, crop_image):
        """核心旋转校正流程"""
        # 红色区域检测
        mask = self._preprocess_red_mask(crop_image)
        
        # 轮廓分析
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop_image
            
        # 最大轮廓处理
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        (_, _), (w, h), angle = rect
        
        # 角度修正逻辑
        if w > h:
            angle += 90
        return self._correct_rotation(crop_image, angle)

    def ocr_recognize(self, image):
        """执行OCR识别"""
        # 预处理增强对比度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 执行OCR
        results = self.reader.readtext(enhanced, detail=0)
        return " ".join(results).upper()

def main():
    # 获取模型路径
    possible_model_dirs = [
        "../weights",
        "weights", 
        "../ready/weights",
        "D:/AirmodelingTeam/CQU_Ground_Recog_Strile_YoloOcr/weights"
    ]
    
    model_dir = None
    for path in possible_model_dirs:
        if os.path.exists(path):
            model_dir = path
            break
    
    if model_dir is None:
        print("未找到模型目录，请检查weights文件夹位置")
        return
    
    pt_model_path = os.path.join(model_dir, "best1.pt")
    trt_model_path = os.path.join(model_dir, "best1_trt.engine")
    
    # 检查模型文件
    if os.path.exists(trt_model_path):
        model_path = trt_model_path
        print(f"使用TensorRT引擎: {trt_model_path}")
    elif os.path.exists(pt_model_path):
        model_path = pt_model_path
        print(f"使用PyTorch模型: {pt_model_path}")
    else:
        print("未找到模型文件")
        return

    # 初始化检测器和处理器
    try:
        detector = YOLOTRTDetector(model_path=model_path, conf_thres=0.25, use_trt=model_path.endswith('.engine'))
        processor = ArrowProcessor()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 视频源设置
    video_sources = [
        0,  # 默认摄像头
        1,  # 第二个摄像头
        "test_video.mp4",  # 测试视频文件
    ]
    
    # 尝试打开视频源
    cap = None
    video_fps = 30.0  # 默认FPS
    is_video_file = False
    
    for source in video_sources:
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # 测试是否能读取帧
                ret, test_frame = cap.read()
                if ret:
                    print(f"成功打开视频源: {source}")
                    
                    # 获取视频FPS
                    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    
                    # 判断是否为视频文件
                    if isinstance(source, str) and source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        is_video_file = True
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        print(f"视频文件 - FPS: {video_fps}, 总帧数: {total_frames}")
                    else:
                        is_video_file = False
                        print(f"摄像头 - FPS: {video_fps}")
                    
                    # 重置到开始
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                else:
                    cap.release()
            else:
                if cap:
                    cap.release()
        except Exception as e:
            print(f"尝试打开视频源 {source} 失败: {e}")
            if cap:
                cap.release()
    
    if cap is None or not cap.isOpened():
        print("无法打开任何视频源")
        return
    
    # 显示设置
    cv2.namedWindow("实时检测", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("实时检测", 1280, 720)
    
    # 实时处理相关变量
    target_fps = video_fps if is_video_file else 30.0  # 目标FPS
    frame_interval = 1.0 / target_fps  # 帧间隔时间
    
    # 统计变量
    last_time = time.time()
    processed_frames = 0
    total_frames_read = 0
    dropped_frames = 0
    process_times = []
    
    print(f"目标FPS: {target_fps:.1f}")
    print("按'q'键退出，按's'键显示统计信息")
    print("实时模式：会自动跳帧以保持实时性")

    try:
        while True:
            current_time = time.time()
            
            # === 实时跳帧逻辑 ===
            if is_video_file:
                # 对于视频文件，计算应该读取到的帧位置
                elapsed_time = current_time - last_time if processed_frames > 0 else 0
                expected_frame_pos = int((time.time() - (last_time - elapsed_time)) * target_fps)
                current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # 如果当前帧位置落后太多，跳帧
                if expected_frame_pos > current_frame_pos + 5:  # 落后超过5帧就跳帧
                    skip_frames = min(expected_frame_pos - current_frame_pos, 30)  # 最多跳30帧
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos + skip_frames)
                    dropped_frames += skip_frames
                    print(f"跳帧 {skip_frames} 帧，当前位置: {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    # 视频结束，重新开始
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_time = time.time()  # 重置时间
                    continue
                else:
                    print("摄像头断开")
                    break
            
            total_frames_read += 1
            
            # === 处理当前帧 ===
            process_start = time.time()
            
            # 调整图像大小以加快处理速度
            scale_percent = 75
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height))

            # YOLO检测
            detections = detector.detect(frame)
            
            # 处理检测结果
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det['box'])
                
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
                    text = processor.ocr_recognize(rotated)
                    
                    # 可视化
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y2 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # 显示预览（简化版）
                    if i == 0:  # 只显示第一个检测结果的预览
                        preview = cv2.resize(rotated, (120, 120))
                        frame[60:180, 10:130] = preview
                    
                except Exception as e:
                    continue
            
            # 计算处理时间
            process_time = time.time() - process_start
            process_times.append(process_time)
            if len(process_times) > 30:
                process_times.pop(0)
            
            processed_frames += 1
            
            # 计算实时FPS
            if processed_frames % 30 == 0:  # 每30帧计算一次
                current_fps = 30 / (current_time - last_time) if processed_frames > 0 else 0
                last_time = current_time
            else:
                current_fps = 0
            
            # 显示性能信息
            avg_process_time = sum(process_times) / len(process_times) if process_times else 0
            
            info_texts = [
                f"处理FPS: {current_fps:.1f}" if current_fps > 0 else "处理FPS: 计算中...",
                f"目标FPS: {target_fps:.1f}",
                f"处理时间: {avg_process_time*1000:.1f}ms",
                f"丢帧数: {dropped_frames}",
                f"总帧数: {total_frames_read}",
                f"模式: {'视频文件' if is_video_file else '摄像头'}"
            ]
            
            for i, text in enumerate(info_texts):
                cv2.putText(frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 显示结果
            cv2.imshow("实时检测", frame)
            
            # === 实时控制 ===
            # 对于摄像头，控制显示帧率
            if not is_video_file:
                elapsed = time.time() - current_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n=== 实时统计 ===")
                print(f"目标FPS: {target_fps:.2f}")
                print(f"当前处理FPS: {current_fps:.2f}")
                print(f"平均处理时间: {avg_process_time*1000:.2f}ms")
                print(f"总读取帧数: {total_frames_read}")
                print(f"处理帧数: {processed_frames}")
                print(f"丢帧数: {dropped_frames}")
                print(f"丢帧率: {dropped_frames/total_frames_read*100:.1f}%" if total_frames_read > 0 else "丢帧率: 0%")

    except KeyboardInterrupt:
        print("\n收到中断信号")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        # 释放资源
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        # 最终统计
        print(f"\n=== 最终统计 ===")
        print(f"总读取帧数: {total_frames_read}")
        print(f"处理帧数: {processed_frames}")
        print(f"丢帧数: {dropped_frames}")
        if total_frames_read > 0:
            print(f"丢帧率: {dropped_frames/total_frames_read*100:.1f}%")
        if process_times:
            print(f"平均处理时间: {sum(process_times)/len(process_times)*1000:.2f}ms")

if __name__ == "__main__":
    main() 