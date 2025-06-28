#!/usr/bin/env python3
# inference4_realtime_tensorrt.py
# TensorRT优化版本 - 针对Jetson Orin Nano优化

import cv2
import numpy as np
import time
import os
import threading
from queue import Queue, Empty
import easyocr
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 添加TensorRT优化检测器路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_trt_utils_optimized import JetsonOptimizedYOLODetector

# 尝试导入Jetson监控工具
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
    print("⚠️  jtop未安装，无法监控Jetson状态")

class FrameBuffer:
    """优化的帧缓冲区"""
    def __init__(self, maxsize=2):  # 减少缓冲区大小以降低延迟
        self.queue = Queue(maxsize=maxsize)
        self.dropped_frames = 0
        self.lock = threading.Lock()
    
    def put_frame(self, frame, timestamp):
        """添加帧到缓冲区"""
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
    
    def get_frame(self, timeout=0.005):  # 减少等待时间
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
    """优化的视频捕获类"""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        
        # 优化摄像头设置
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            # 设置较低分辨率以提高性能
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.frame_buffer = FrameBuffer(maxsize=2)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_interval = 1.0 / self.fps
        self.capture_thread = None
        self.running = False
        self.total_frames = 0
        self.start_time = None
        
        # 视频文件处理
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
        """优化的帧捕获线程"""
        last_frame_time = time.time()
        frame_skip_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            current_time = time.time()
            self.total_frames += 1
            
            # 动态跳帧策略
            if self.is_video_file:
                elapsed_time = current_time - self.start_time
                expected_frame = int(elapsed_time * self.fps)
                
                if self.total_frames < expected_frame:
                    skip_frames = min(expected_frame - self.total_frames, 5)
                    for _ in range(skip_frames):
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                        self.total_frames += 1
                        frame_skip_count += 1
            
            # 添加帧到缓冲区
            self.frame_buffer.put_frame(frame, current_time)
            
            # 控制捕获频率
            if not self.is_video_file:
                sleep_time = max(0, self.frame_interval - (current_time - last_frame_time))
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

class AsyncOCRProcessor:
    """异步OCR处理器"""
    def __init__(self, max_workers=2):
        print("🔤 初始化异步OCR处理器...")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        self.ocr_cache = {}  # 简单的OCR结果缓存
        self.cache_timeout = 2.0  # 缓存超时时间
        
    def _ocr_recognize(self, image_key, image):
        """执行OCR识别"""
        # 检查缓存
        current_time = time.time()
        if image_key in self.ocr_cache:
            result, timestamp = self.ocr_cache[image_key]
            if current_time - timestamp < self.cache_timeout:
                return result
        
        # 预处理增强对比度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 执行OCR
        try:
            results = self.reader.readtext(enhanced, detail=0)
            text = " ".join(results).upper()
            
            # 更新缓存
            self.ocr_cache[image_key] = (text, current_time)
            
            # 清理过期缓存
            if len(self.ocr_cache) > 20:
                self._cleanup_cache()
            
            return text
        except Exception as e:
            print(f"OCR错误: {e}")
            return ""
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.ocr_cache.items()
            if current_time - timestamp > self.cache_timeout
        ]
        for key in expired_keys:
            del self.ocr_cache[key]
    
    def process_async(self, image_key, image):
        """异步处理OCR"""
        return self.executor.submit(self._ocr_recognize, image_key, image)
    
    def shutdown(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)

class ArrowProcessor:
    """优化的箭头处理器"""
    def __init__(self):
        # 初始化异步OCR
        self.ocr_processor = AsyncOCRProcessor(max_workers=1)  # Jetson上使用单线程OCR
        
        # 红色阈值范围（HSV颜色空间）
        self.lower_red1 = np.array([0, 30, 30])
        self.lower_red2 = np.array([150, 30, 30])
        self.upper_red1 = np.array([30, 255, 255])
        self.upper_red2 = np.array([179, 255, 255])
        
        # 优化的形态学处理核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # OCR任务管理
        self.ocr_futures = {}
        self.ocr_results = {}

    def _preprocess_red_mask(self, image):
        """优化的红色区域预处理"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        combined = cv2.bitwise_or(mask1, mask2)
        
        # 减少形态学操作以提高速度
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return cleaned

    def _correct_rotation(self, image, angle):
        """优化的旋转校正"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
        
        # 简化的方向验证
        rotated_mask = self._preprocess_red_mask(rotated)
        top = rotated_mask[:h//2, :]
        bottom = rotated_mask[h//2:, :]
        if cv2.countNonZero(bottom) > cv2.countNonZero(top):
            rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        return rotated

    def rotate_arrow(self, crop_image):
        """优化的旋转校正流程"""
        # 检查图像大小，如果太小就跳过处理
        if crop_image.shape[0] < 50 or crop_image.shape[1] < 50:
            return crop_image
            
        mask = self._preprocess_red_mask(crop_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop_image
            
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        (_, _), (w, h), angle = rect
        
        if w > h:
            angle += 90
        return self._correct_rotation(crop_image, angle)

    def process_detection_async(self, detection_id, crop_image):
        """异步处理检测结果"""
        # 旋转校正
        rotated = self.rotate_arrow(crop_image)
        
        # 提交OCR任务
        future = self.ocr_processor.process_async(detection_id, rotated)
        self.ocr_futures[detection_id] = future
        
        return rotated

    def get_ocr_result(self, detection_id):
        """获取OCR结果"""
        if detection_id in self.ocr_futures:
            future = self.ocr_futures[detection_id]
            if future.done():
                try:
                    result = future.result()
                    self.ocr_results[detection_id] = result
                    del self.ocr_futures[detection_id]
                    return result
                except Exception as e:
                    print(f"OCR任务失败: {e}")
                    del self.ocr_futures[detection_id]
                    return ""
        
        return self.ocr_results.get(detection_id, "处理中...")

    def cleanup_old_results(self, active_detection_ids):
        """清理不再需要的OCR结果"""
        # 清理完成的结果
        for detection_id in list(self.ocr_results.keys()):
            if detection_id not in active_detection_ids:
                del self.ocr_results[detection_id]
        
        # 清理未完成的任务
        for detection_id in list(self.ocr_futures.keys()):
            if detection_id not in active_detection_ids:
                self.ocr_futures[detection_id].cancel()
                del self.ocr_futures[detection_id]

    def shutdown(self):
        """关闭处理器"""
        self.ocr_processor.shutdown()

class JetsonMonitor:
    """Jetson性能监控器"""
    def __init__(self):
        self.jetson_available = JTOP_AVAILABLE
        self.stats = {}
        
    def get_stats(self):
        """获取Jetson状态"""
        if not self.jetson_available:
            return {}
        
        try:
            with jtop() as jetson:
                if jetson.ok():
                    self.stats = {
                        'gpu_usage': jetson.gpu.get('GR3D', {}).get('val', 0),
                        'cpu_usage': sum(jetson.cpu.values()) / len(jetson.cpu),
                        'memory_used': jetson.memory['RAM']['used'],
                        'memory_total': jetson.memory['RAM']['tot'],
                        'temperature': jetson.temperature.get('CPU', 0),
                        'power': jetson.power.get('cur', 0),
                        'nvpmodel': jetson.nvpmodel.get('name', 'Unknown')
                    }
        except Exception as e:
            print(f"Jetson监控错误: {e}")
        
        return self.stats

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
        print("❌ 未找到模型目录，请检查weights文件夹位置")
        return
    
    # 优先使用TensorRT引擎
    trt_model_path = os.path.join(model_dir, "best1.engine")
    pt_model_path = os.path.join(model_dir, "best1.pt")
    
    if os.path.exists(trt_model_path):
        model_path = trt_model_path
        print(f"🚀 使用TensorRT引擎: {trt_model_path}")
    elif os.path.exists(pt_model_path):
        model_path = pt_model_path
        print(f"📦 使用PyTorch模型: {pt_model_path}")
    else:
        print("❌ 未找到模型文件")
        return

    # 初始化组件
    try:
        print("🔧 初始化TensorRT优化检测器...")
        detector = JetsonOptimizedYOLODetector(model_path=model_path, conf_thres=0.25)
        
        print("🔄 初始化箭头处理器...")
        processor = ArrowProcessor()
        
        print("📊 初始化Jetson监控器...")
        jetson_monitor = JetsonMonitor()
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 视频源设置
    video_sources = [
        0,  # 默认摄像头
        1,  # 第二个摄像头
        "test_video.mp4",  # 测试视频文件
    ]
    
    # 尝试打开视频源
    video_capture = None
    for source in video_sources:
        try:
            temp_cap = VideoCapture(source)
            if temp_cap.start():
                video_capture = temp_cap
                print(f"📹 成功打开视频源: {source}")
                break
            else:
                temp_cap.stop()
        except Exception as e:
            print(f"⚠️  尝试打开视频源 {source} 失败: {e}")
    
    if video_capture is None:
        print("❌ 无法打开任何视频源")
        return
    
    # 显示设置
    cv2.namedWindow("TensorRT实时检测", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TensorRT实时检测", 1280, 720)
    
    # 性能统计
    process_times = []
    last_stats_time = time.time()
    processed_frames = 0
    detection_id_counter = 0
    
    print("🎯 TensorRT优化版本启动完成!")
    print("按'q'键退出，按's'键显示详细统计信息，按'j'键显示Jetson状态")

    try:
        while True:
            # 获取最新帧
            frame, timestamp = video_capture.read()
            if frame is None:
                time.sleep(0.005)  # 减少等待时间
                continue
            
            process_start = time.time()
            
            # 动态调整图像大小（根据性能调整）
            avg_process_time = sum(process_times[-10:]) / len(process_times[-10:]) if process_times else 0
            if avg_process_time > 0.1:  # 如果处理时间超过100ms
                scale_percent = 60  # 降低分辨率
            else:
                scale_percent = 75  # 正常分辨率
            
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height))

            # YOLO检测
            detections = detector.detect(frame)
            
            # 处理检测结果
            preview_height = 100
            preview_width = 100
            spacing = 10
            active_detection_ids = []
            
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det['box'])
                detection_id = f"det_{detection_id_counter}_{i}"
                active_detection_ids.append(detection_id)
                
                try:
                    # 扩展检测框（减少扩展比例以提高性能）
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
                    
                    # 异步处理旋转和OCR
                    rotated = processor.process_detection_async(detection_id, crop)
                    
                    # 获取OCR结果（可能是缓存的或正在处理的）
                    text = processor.get_ocr_result(detection_id)
                    
                    # 可视化
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{text} ({det['confidence']:.2f})", 
                               (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 显示预览（减少预览大小）
                    if i < 5:  # 最多显示5个预览
                        preview = cv2.resize(rotated, (preview_width, preview_height))
                        x_offset = 10 + i * (preview_width + spacing)
                        y_offset = 10
                        if x_offset + preview_width <= frame.shape[1] and y_offset + preview_height <= frame.shape[0]:
                            frame[y_offset:y_offset + preview_height, x_offset:x_offset + preview_width] = preview
                    
                except Exception as e:
                    continue
            
            # 清理旧的OCR结果
            processor.cleanup_old_results(active_detection_ids)
            detection_id_counter += 1
            
            # 计算处理时间
            process_time = time.time() - process_start
            process_times.append(process_time)
            if len(process_times) > 30:
                process_times.pop(0)
            
            processed_frames += 1
            
            # 获取统计信息
            stats = video_capture.get_stats()
            detector_stats = detector.get_performance_stats()
            jetson_stats = jetson_monitor.get_stats()
            
            # 显示性能信息
            current_time = time.time()
            if current_time - last_stats_time >= 1.0:
                processing_fps = processed_frames / (current_time - last_stats_time)
                processed_frames = 0
                last_stats_time = current_time
            else:
                processing_fps = 0
            
            avg_process_time = sum(process_times) / len(process_times) if process_times else 0
            
            # 显示信息
            info_texts = [
                f"🚀 TensorRT: {detector.using_trt}",
                f"📊 处理FPS: {processing_fps:.1f}",
                f"🎥 视频FPS: {stats['actual_fps']:.1f}/{stats['target_fps']:.1f}",
                f"⏱️  处理时间: {avg_process_time*1000:.1f}ms",
                f"📉 丢帧: {stats['dropped_frames']}",
            ]
            
            # 添加Jetson状态信息
            if jetson_stats:
                info_texts.extend([
                    f"🔥 GPU: {jetson_stats.get('gpu_usage', 0):.1f}%",
                    f"🌡️  温度: {jetson_stats.get('temperature', 0):.1f}°C",
                    f"⚡ 功耗: {jetson_stats.get('power', 0):.1f}W"
                ])
            
            for i, text in enumerate(info_texts):
                cv2.putText(frame, text, (10, 150 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 显示结果
            cv2.imshow("TensorRT实时检测", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n=== 详细统计信息 ===")
                print(f"视频FPS: {stats['actual_fps']:.2f}/{stats['target_fps']:.2f}")
                print(f"处理FPS: {processing_fps:.2f}")
                print(f"平均处理时间: {avg_process_time*1000:.2f}ms")
                print(f"TensorRT状态: {detector.using_trt}")
                print(f"检测器统计: {detector_stats}")
            elif key == ord('j') and jetson_stats:
                print(f"\n=== Jetson状态 ===")
                for key, value in jetson_stats.items():
                    print(f"{key}: {value}")

    except KeyboardInterrupt:
        print("\n⏹️  收到中断信号")
    except Exception as e:
        print(f"❌ 运行时错误: {e}")
    finally:
        # 释放资源
        print("🧹 清理资源中...")
        if video_capture:
            video_capture.stop()
        processor.shutdown()
        cv2.destroyAllWindows()
        
        # 最终统计
        final_stats = video_capture.get_stats() if video_capture else {}
        final_detector_stats = detector.get_performance_stats()
        print(f"\n=== 最终统计 ===")
        print(f"总处理帧数: {final_stats.get('total_frames', 0)}")
        print(f"平均视频FPS: {final_stats.get('actual_fps', 0):.2f}")
        print(f"TensorRT平均FPS: {final_detector_stats.get('fps', 0):.2f}")
        if process_times:
            print(f"平均处理时间: {sum(process_times)/len(process_times)*1000:.2f}ms")

if __name__ == "__main__":
    main() 