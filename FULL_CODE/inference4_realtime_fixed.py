#!/usr/bin/env python3
# inference4_realtime_fixed.py
# 修复段错误问题的实时推理脚本

import cv2
import numpy as np
import time
import os
import threading
from queue import Queue, Empty
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

# 延迟导入可能引起段错误的模块
def safe_import_easyocr():
    """安全导入EasyOCR"""
    try:
        import easyocr
        return easyocr
    except Exception as e:
        print(f"EasyOCR导入失败: {e}")
        return None

def safe_import_yolo():
    """安全导入YOLO检测器"""
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception as e:
        print(f"YOLO导入失败: {e}")
        return None

class FrameBuffer:
    """帧缓冲区 - 用于实时视频处理"""
    def __init__(self, maxsize=2):  # 减少缓冲区大小
        self.queue = Queue(maxsize=maxsize)
        self.dropped_frames = 0
        self.lock = threading.Lock()
    
    def put_frame(self, frame, timestamp):
        """添加帧到缓冲区，如果满了则丢弃旧帧"""
        with self.lock:
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
        with self.lock:
            count = self.dropped_frames
            self.dropped_frames = 0
            return count

class VideoCapture:
    """实时视频捕获类"""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.frame_buffer = FrameBuffer(maxsize=2)  # 减少缓存
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_interval = 1.0 / self.fps
        self.capture_thread = None
        self.running = False
        self.total_frames = 0
        self.start_time = None
        
        # 设置较低的分辨率以减少内存使用
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
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
            try:
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
                
                # 将帧添加到缓冲区
                self.frame_buffer.put_frame(frame, current_time)
                
                # 控制捕获频率
                if not self.is_video_file:
                    sleep_time = self.frame_interval - (current_time - last_frame_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                last_frame_time = current_time
                
            except Exception as e:
                print(f"捕获帧时出错: {e}")
                time.sleep(0.1)
    
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

class SafeYOLODetector:
    """安全的YOLO检测器"""
    
    def __init__(self, model_path, conf_thres=0.25):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.model = None
        self.using_trt = model_path.endswith('.engine')
        
        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 安全加载模型
        self._load_model()
    
    def _load_model(self):
        """安全加载模型"""
        try:
            YOLO = safe_import_yolo()
            if YOLO is None:
                raise RuntimeError("无法导入YOLO")
            
            print(f"正在加载模型: {self.model_path}")
            
            # 设置环境变量以避免CUDA相关问题
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用CPU
            
            self.model = YOLO(self.model_path)
            
            # 强制设置为CPU模式
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            
            print(f"模型加载成功 (CPU模式)")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def detect(self, frame):
        """安全检测"""
        if self.model is None:
            return []
        
        try:
            # 确保输入是numpy数组
            if not isinstance(frame, np.ndarray):
                return []
            
            # 调整图像大小以加快处理速度
            if frame.shape[0] > 480 or frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, 480))
            
            # 执行检测
            results = self.model.predict(
                frame, 
                conf=self.conf_thres, 
                verbose=False,
                device='cpu'  # 强制使用CPU
            )
            
            detections = []
            
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                            conf = float(box.conf[0].cpu().item())
                            cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else 0
                            
                            detections.append({
                                'box': (x1, y1, x2, y2),
                                'confidence': conf,
                                'class_id': cls_id
                            })
                        except Exception as e:
                            continue
            
            return detections
            
        except Exception as e:
            print(f"检测过程中出现错误: {e}")
            return []

class SimpleArrowProcessor:
    """简化的箭头处理器 - 避免EasyOCR段错误"""
    
    def __init__(self):
        self.reader = None
        self.ocr_available = False
        
        # 尝试安全初始化OCR
        self._init_ocr()
        
        # 红色阈值范围（HSV颜色空间）
        self.lower_red1 = np.array([0, 30, 30])
        self.lower_red2 = np.array([150, 30, 30])
        self.upper_red1 = np.array([30, 255, 255])
        self.upper_red2 = np.array([179, 255, 255])
        
        # 形态学处理核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    def _init_ocr(self):
        """安全初始化OCR"""
        try:
            easyocr = safe_import_easyocr()
            if easyocr is not None:
                print("正在初始化EasyOCR (CPU模式)...")
                self.reader = easyocr.Reader(['en'], gpu=False, download_enabled=False)
                self.ocr_available = True
                print("EasyOCR初始化成功")
            else:
                print("EasyOCR不可用，将跳过文字识别")
        except Exception as e:
            print(f"OCR初始化失败: {e}")
            self.ocr_available = False
    
    def rotate_arrow(self, crop_image):
        """简化的旋转校正"""
        try:
            # 红色区域检测
            hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            combined = cv2.bitwise_or(mask1, mask2)
            
            # 形态学处理
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel, iterations=1)
            
            # 轮廓分析
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return crop_image
            
            # 最大轮廓处理
            max_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(max_contour)
            (_, _), (w, h), angle = rect
            
            # 简化的角度修正
            if w > h:
                angle += 90
            
            # 执行旋转
            (img_h, img_w) = crop_image.shape[:2]
            center = (img_w // 2, img_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(crop_image, M, (img_w, img_h), borderValue=(255, 255, 255))
            
            return rotated
            
        except Exception as e:
            print(f"旋转处理出错: {e}")
            return crop_image
    
    def ocr_recognize(self, image):
        """安全的OCR识别"""
        if not self.ocr_available or self.reader is None:
            return "OCR不可用"
        
        try:
            # 预处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 简单的对比度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # 执行OCR
            results = self.reader.readtext(enhanced, detail=0)
            return " ".join(results).upper() if results else "无文字"
            
        except Exception as e:
            print(f"OCR识别出错: {e}")
            return "识别失败"

def main():
    print("=== 修复版实时检测系统 ===")
    
    # 获取模型路径
    possible_model_dirs = [
        "../weights",
        "weights", 
        "../ready/weights",
        "/home/lyc/CQU_Ground_ReconnaissanceStrike/weights"
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
    
    # 检查模型文件
    if not os.path.exists(pt_model_path):
        print(f"未找到模型文件: {pt_model_path}")
        return
    
    # 初始化组件
    try:
        print("初始化检测器...")
        detector = SafeYOLODetector(pt_model_path, conf_thres=0.25)
        
        print("初始化箭头处理器...")
        processor = SimpleArrowProcessor()
        
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 视频源设置
    video_sources = [
        0,  # 默认摄像头
        1,  # 第二个摄像头
    ]
    
    # 尝试打开视频源
    video_capture = None
    for source in video_sources:
        try:
            print(f"尝试打开视频源: {source}")
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
    cv2.resizeWindow("实时检测", 800, 600)
    
    # 性能统计
    process_times = []
    last_stats_time = time.time()
    processed_frames = 0
    
    print("按'q'键退出，按's'键显示统计信息")
    print("系统已启动，正在处理...")

    try:
        while True:
            # 获取最新帧
            frame, timestamp = video_capture.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # 处理开始计时
            process_start = time.time()
            
            # 调整图像大小
            if frame.shape[0] > 480 or frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, 480))

            # YOLO检测
            detections = detector.detect(frame)
            
            # 处理检测结果
            for i, det in enumerate(detections):
                try:
                    x1, y1, x2, y2 = map(int, det['box'])
                    
                    # 扩展检测框（减少扩展比例）
                    expand_ratio = 0.05
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
                    cv2.putText(frame, text[:20], (x1, y2 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                except Exception as e:
                    print(f"处理检测结果时出错: {e}")
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
                f"视频FPS: {stats['actual_fps']:.1f}",
                f"处理时间: {avg_process_time*1000:.1f}ms",
                f"丢帧: {stats['dropped_frames']}",
                f"模式: CPU"
            ]
            
            for i, text in enumerate(info_texts):
                cv2.putText(frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 显示结果
            cv2.imshow("实时检测", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n=== 统计信息 ===")
                print(f"视频FPS: {stats['actual_fps']:.2f}")
                print(f"处理FPS: {processing_fps:.2f}")
                print(f"平均处理时间: {avg_process_time*1000:.2f}ms")
                print(f"总帧数: {stats['total_frames']}")
                print(f"丢帧数: {stats['dropped_frames']}")
            
            # 定期清理内存
            if processed_frames % 100 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("\n收到中断信号")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        # 释放资源
        if video_capture:
            video_capture.stop()
        cv2.destroyAllWindows()
        
        print(f"\n=== 程序结束 ===")

if __name__ == "__main__":
    main() 