#!/usr/bin/env python3
# camera_inference4.py
# 基于camera_inference3.py，但使用TensorRT加速YOLO推理

import cv2
import numpy as np
import time
import os
from yolo_trt_utils import YOLOTRTDetector  # 导入TensorRT支持的YOLO检测器
import easyocr

class ArrowProcessor:
    def __init__(self):
        # 初始化OCR（全局单例）
        # 在Windows平台使用CPU模式以确保兼容性
        print("EasyOCR使用CPU模式")
        self.reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        
        # 红色阈值范围（HSV颜色空间）
        self.lower_red1 = np.array([0, 30, 30])
        self.lower_red2 = np.array([150, 30, 30])
        self.upper_red1 = np.array([30, 255, 255])
        self.upper_red2 = np.array([179, 255, 255])  # 修正 upper_red2 的 H 值
        
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
            return crop_image  # 无轮廓时退回原图
            
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
        return " ".join(results).upper()  # 返回合并后的大写字符串

def main():
    # 获取模型路径 - 适应Windows环境
    # 首先尝试相对路径
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
        print("尝试的路径：", possible_model_dirs)
        return
    
    pt_model_path = os.path.join(model_dir, "best.pt")
    trt_model_path = os.path.join(model_dir, "best_trt.engine")
    
    # 检查TensorRT引擎是否存在，否则使用原始模型
    if os.path.exists(trt_model_path):
        model_path = trt_model_path
        print(f"使用TensorRT引擎: {trt_model_path}")
    elif os.path.exists(pt_model_path):
        model_path = pt_model_path
        print(f"使用PyTorch模型: {pt_model_path}")
    else:
        print(f"未找到模型文件，请确保存在以下文件之一:")
        print(f"  - {pt_model_path}")
        print(f"  - {trt_model_path}")
        return

    # 初始化检测器和处理器
    try:
        detector = YOLOTRTDetector(model_path=model_path, conf_thres=0.25, use_trt=model_path.endswith('.engine'))
        processor = ArrowProcessor()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 视频输入设置 - 优先使用摄像头
    video_sources = [
        0,  # 默认摄像头
        1,  # 第二个摄像头
        "test_video.mp4",  # 测试视频文件
    ]
    
    cap = None
    for source in video_sources:
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # 测试是否能读取帧
                ret, test_frame = cap.read()
                if ret:
                    print(f"成功打开视频源: {source}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
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
    cv2.namedWindow("TensorRT Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TensorRT Detection", 1280, 720)
    
    # 性能统计
    frame_count = 0
    fps_avg = 0
    process_times = []  # 用于计算处理时间
    start_time = time.time()
    
    print("按'q'键退出")

    try:
        while True:
            # 计时开始
            loop_start = time.time()
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("视频结束或无法读取帧")
                break
                
            # 调整图像大小以加快处理速度
            scale_percent = 75  # 缩小到原始尺寸的75%
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height))

            # YOLO检测 - 使用TensorRT加速
            detections = detector.detect(frame)
            
            # 显示设置
            preview_height = 120
            preview_width = 120
            spacing = 10  # 图像间距
            
            # 计算当前FPS
            frame_count += 1
            current_time = time.time()
            if current_time - start_time >= 1.0:  # 每秒更新一次FPS
                fps_avg = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time
            
            # 处理检测结果
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det['box'])
                
                try:
                    # 扩展比例
                    expand_ratio = 0.1
                    
                    # 计算原始检测框的宽度和高度
                    width_det = x2 - x1
                    height_det = y2 - y1
                    
                    # 计算扩展量
                    expand_w = int(width_det * expand_ratio)
                    expand_h = int(height_det * expand_ratio)
                    
                    # 计算扩展后的裁剪坐标，确保不超出图像边界
                    x1_exp = max(0, x1 - expand_w)
                    y1_exp = max(0, y1 - expand_h)
                    x2_exp = min(frame.shape[1], x2 + expand_w)
                    y2_exp = min(frame.shape[0], y2 + expand_h)
                    
                    # 裁剪扩展后的区域
                    crop = frame[y1_exp:y2_exp, x1_exp:x2_exp].copy()
                    if crop.size == 0:
                        continue
                    
                    # 旋转校正
                    rotated = processor.rotate_arrow(crop)
                    
                    # OCR识别
                    text = processor.ocr_recognize(rotated)
                    
                    # 可视化检测框和OCR结果
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y2 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # 动态显示旋转校正后的图像
                    preview = cv2.resize(rotated, (preview_width, preview_height))
                    x_offset = 10 + i * (preview_width + spacing)
                    y_offset = 60  # 避开FPS显示
                    # 确保预览区域不超出帧边界
                    if x_offset + preview_width <= frame.shape[1] and y_offset + preview_height <= frame.shape[0]:
                        frame[y_offset:y_offset + preview_height, x_offset:x_offset + preview_width] = preview
                    
                except Exception as e:
                    print(f"处理异常: {str(e)}")
                    continue
            
            # 计算单帧处理时间
            process_time = time.time() - loop_start
            process_times.append(process_time)
            if len(process_times) > 30:  # 保持最近30帧的数据
                process_times.pop(0)
            avg_process_time = sum(process_times) / len(process_times)
            
            # 显示FPS和处理时间
            fps_text = f"FPS: {fps_avg:.1f}"
            time_text = f"处理时间: {avg_process_time*1000:.1f} ms"
            trt_text = f"模式: {'TensorRT' if detector.using_trt else 'PyTorch'}"
            
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, trt_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示结果
            cv2.imshow("TensorRT Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n收到中断信号")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        # 释放资源
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        # 打印性能统计
        print(f"\n性能统计:")
        print(f"平均FPS: {fps_avg:.2f}")
        if process_times:
            print(f"平均处理时间: {sum(process_times)/len(process_times)*1000:.2f} ms")
        print(f"加速模式: {'TensorRT' if detector.using_trt else 'PyTorch'}")

if __name__ == "__main__":
    main() 