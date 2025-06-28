import cv2
import numpy as np
import time
import os
import threading
from queue import Queue, Empty
from yolo_trt_utils import YOLOTRTDetector  # 导入TensorRT支持的YOLO检测器
import easyocr

class FrameBuffer:
    """帧缓冲区 - 用于实时视频处理"""
    def __init__(self, maxsize=3):
        self.queue = Queue(maxsize=maxsize)
        self.dropped_frames = 0
    
    def put_frame(self, frame, timestamp):
        """添加帧到缓冲区，如果满了则丢弃旧帧"""
        try:
            # 如果队列满了，先清空再添加新帧
            while self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.dropped_frames += 1
                except Empty:
                    break
            
            self.queue.put((frame, timestamp), block=False)
        except:
            self.dropped_frames += 1
    
    def get_frame(self, timeout=0.01):
        """获取最新帧"""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None, None
    
    def get_dropped_count(self):
        """获取丢帧数量"""
        count = self.dropped_frames
        self.dropped_frames = 0
        return count

class VideoCapture:
    """实时视频捕获类"""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.frame_buffer = FrameBuffer(maxsize=3)  # 最多缓存3帧
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_interval = 1.0 / self.fps
        self.capture_thread = None
        self.running = False
        self.total_frames = 0
        self.start_time = None
        
        # 如果是视频文件，获取总帧数
        if isinstance(source, str) and source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.is_video_file = True
        else:
            self.total_video_frames = 0
            self.is_video_file = False
    
    def start(self):
        """开始捕获"""
        if self.cap.isOpened():
            self.running = True
            self.start_time = time.time()
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            return True
        return False
    
    def _capture_frames(self):
        """捕获帧的线程函数"""
        last_frame_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    # 视频文件结束，循环播放
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # 摄像头断开
                    break
            
            current_time = time.time()
            self.total_frames += 1
            
            # 对于视频文件，按照原始FPS进行时间控制
            if self.is_video_file:
                elapsed_time = current_time - self.start_time
                expected_frame = int(elapsed_time * self.fps)
                
                # 如果处理速度跟不上，跳帧到当前时间应该的帧
                if self.total_frames < expected_frame:
                    skip_frames = expected_frame - self.total_frames
                    for _ in range(min(skip_frames, 10)):  # 最多一次跳10帧
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                        self.total_frames += 1
            
            # 将帧添加到缓冲区
            self.frame_buffer.put_frame(frame, current_time)
            
            # 控制捕获频率（主要针对摄像头）
            if not self.is_video_file:
                sleep_time = self.frame_interval - (current_time - last_frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            last_frame_time = current_time
    
    def read(self):
        """读取最新帧"""
        return self.frame_buffer.get_frame()
    
    def get_stats(self):
        """获取统计信息"""
        dropped = self.frame_buffer.get_dropped_count()
        if self.start_time:
            elapsed = time.time() - self.start_time
            actual_fps = self.total_frames / elapsed if elapsed > 0 else 0
        else:
            actual_fps = 0
        
        return {
            'dropped_frames': dropped,
            'total_frames': self.total_frames,
            'actual_fps': actual_fps,
            'target_fps': self.fps
        }
    
    def stop(self):
        """停止捕获"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

class ArrowProcessor:
    def __init__(self):
        # 初始化OCR（全局单例）
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
    
    pt_model_path = os.path.join(model_dir, "best.pt")
    trt_model_path = os.path.join(model_dir, "best_trt.engine")
    
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
        "/home/lyc/CQU_Ground_ReconnaissanceStrike/video1.mp4",  # 测试视频文件
    ]
    
    # 尝试打开视频源
    video_capture = None
    for source in video_sources:
        try:
            temp_cap = VideoCapture(source)
            if temp_cap.start():
                video_capture = temp_cap
                print(f"成功打开视频源: {source}")
                break
            else:
                temp_cap.stop()
        except Exception as e:
            print(f"尝试打开视频源 {source} 失败: {e}")
    
    if video_capture is None:
        print("无法打开任何视频源")
        return
    
    # 显示设置
    cv2.namedWindow("实时检测", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("实时检测", 1280, 720)
    
    # 性能统计
    process_times = []
    last_stats_time = time.time()
    processed_frames = 0
    
    print("按'q'键退出，按's'键显示统计信息")

    try:
        while True:
            # 获取最新帧
            frame, timestamp = video_capture.read()
            if frame is None:
                time.sleep(0.01)  # 短暂等待
                continue
            
            # 处理开始计时
            process_start = time.time()
            
            # 调整图像大小以加快处理速度
            scale_percent = 75
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height))

            # YOLO检测
            detections = detector.detect(frame)
            
            # 处理检测结果
            preview_height = 120
            preview_width = 120
            spacing = 10
            
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
                    
                    # 显示预览
                    preview = cv2.resize(rotated, (preview_width, preview_height))
                    x_offset = 10 + i * (preview_width + spacing)
                    y_offset = 90
                    if x_offset + preview_width <= frame.shape[1] and y_offset + preview_height <= frame.shape[0]:
                        frame[y_offset:y_offset + preview_height, x_offset:x_offset + preview_width] = preview
                    
                except Exception as e:
                    continue
            
            # 计算处理时间
            process_time = time.time() - process_start
            process_times.append(process_time)
            if len(process_times) > 30:
                process_times.pop(0)
            
            processed_frames += 1
            
            # 获取视频统计
            stats = video_capture.get_stats()
            
            # 显示性能信息
            current_time = time.time()
            if current_time - last_stats_time >= 1.0:
                processing_fps = processed_frames / (current_time - last_stats_time)
                avg_process_time = sum(process_times) / len(process_times) if process_times else 0
                
                # 重置计数器
                processed_frames = 0
                last_stats_time = current_time
            else:
                processing_fps = 0
                avg_process_time = sum(process_times) / len(process_times) if process_times else 0
            
            # 显示信息
            info_texts = [
                f"处理FPS: {processing_fps:.1f}",
                f"视频FPS: {stats['actual_fps']:.1f}/{stats['target_fps']:.1f}",
                f"处理时间: {avg_process_time*1000:.1f}ms",
                f"丢帧: {stats['dropped_frames']}",
                f"模式: {'TensorRT' if detector.using_trt else 'PyTorch'}"
            ]
            
            for i, text in enumerate(info_texts):
                cv2.putText(frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 显示结果
            cv2.imshow("实时检测", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n=== 统计信息 ===")
                print(f"视频FPS: {stats['actual_fps']:.2f}/{stats['target_fps']:.2f}")
                print(f"处理FPS: {processing_fps:.2f}")
                print(f"平均处理时间: {avg_process_time*1000:.2f}ms")
                print(f"总帧数: {stats['total_frames']}")
                print(f"丢帧数: {stats['dropped_frames']}")

    except KeyboardInterrupt:
        print("\n收到中断信号")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        # 释放资源
        if video_capture:
            video_capture.stop()
        cv2.destroyAllWindows()
        
        # 最终统计
        final_stats = video_capture.get_stats() if video_capture else {}
        print(f"\n=== 最终统计 ===")
        print(f"总处理帧数: {final_stats.get('total_frames', 0)}")
        print(f"平均视频FPS: {final_stats.get('actual_fps', 0):.2f}")
        if process_times:
            print(f"平均处理时间: {sum(process_times)/len(process_times)*1000:.2f}ms")

if __name__ == "__main__":
    main() 