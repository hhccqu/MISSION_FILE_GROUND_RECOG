#!/usr/bin/env python3
# inference4_jetson_optimized.py
# 针对Jetson Orin Nano的高性能优化版本

import cv2
import numpy as np
import time
import os
import threading
from queue import Queue, Empty
from yolo_trt_utils import YOLOTRTDetector
import easyocr

class OptimizedArrowProcessor:
    def __init__(self):
        print("初始化优化的箭头处理器...")
        
        # 使用GPU加速的EasyOCR（如果可用）
        try:
            print("尝试使用GPU模式的EasyOCR...")
            self.reader = easyocr.Reader(['en'], gpu=True, download_enabled=True)
            print("✓ EasyOCR GPU模式启用成功")
        except:
            print("GPU模式失败，回退到CPU模式")
            self.reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        
        # 优化的红色阈值范围
        self.lower_red1 = np.array([0, 50, 50])  # 提高阈值以减少噪声
        self.lower_red2 = np.array([160, 50, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.upper_red2 = np.array([179, 255, 255])
        
        # 更小的形态学核以加速处理
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # OCR缓存机制
        self.ocr_cache = {}
        self.cache_max_size = 10

    def _preprocess_red_mask_fast(self, image):
        """快速红色区域预处理"""
        # 直接在HSV空间操作，减少转换
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 使用更高效的阈值操作
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        combined = cv2.bitwise_or(mask1, mask2)
        
        # 减少形态学操作的迭代次数
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return cleaned

    def _simple_rotation(self, image, angle):
        """简化的旋转操作"""
        if abs(angle) < 5:  # 角度太小则跳过旋转
            return image
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 使用更快的旋转方法
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        return rotated

    def rotate_arrow_fast(self, crop_image):
        """快速旋转校正"""
        # 如果图像太小，直接返回
        if crop_image.shape[0] < 30 or crop_image.shape[1] < 30:
            return crop_image
            
        # 快速红色区域检测
        mask = self._preprocess_red_mask_fast(crop_image)
        
        # 快速轮廓分析
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop_image
            
        # 只处理最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) < 50:  # 轮廓太小则跳过
            return crop_image
            
        rect = cv2.minAreaRect(max_contour)
        (_, _), (w, h), angle = rect
        
        # 简化角度计算
        if w > h:
            angle += 90
            
        return self._simple_rotation(crop_image, angle)

    def ocr_recognize_fast(self, image):
        """快速OCR识别，带缓存"""
        # 计算图像哈希作为缓存键
        image_hash = hash(image.tobytes())
        
        # 检查缓存
        if image_hash in self.ocr_cache:
            return self.ocr_cache[image_hash]
        
        # 快速预处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 简化的对比度增强
        enhanced = cv2.equalizeHist(gray)
        
        # OCR识别
        try:
            results = self.reader.readtext(enhanced, detail=0, paragraph=False, width_ths=0.9)
            text = " ".join(results).upper()
        except:
            text = ""
        
        # 更新缓存
        if len(self.ocr_cache) >= self.cache_max_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.ocr_cache))
            del self.ocr_cache[oldest_key]
        
        self.ocr_cache[image_hash] = text
        return text

class FastVideoCapture:
    """高性能视频捕获类"""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        
        # 优化视频捕获参数
        if self.cap.isOpened():
            # 设置缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 如果是摄像头，设置较低的分辨率以提高速度
            if isinstance(source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_buffer = Queue(maxsize=2)  # 小缓冲区
        self.capture_thread = None
        self.running = False
        self.total_frames = 0
        self.dropped_frames = 0
        
        # 判断是否为视频文件
        if isinstance(source, str) and source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.is_video_file = True
        else:
            self.is_video_file = False

    def start(self):
        """开始捕获"""
        if self.cap.isOpened():
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            return True
        return False

    def _capture_frames(self):
        """捕获帧的线程函数"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            self.total_frames += 1
            
            # 非阻塞放入队列
            try:
                # 如果队列满了，丢弃旧帧
                while self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                        self.dropped_frames += 1
                    except Empty:
                        break
                
                self.frame_buffer.put(frame, block=False)
            except:
                self.dropped_frames += 1

    def read(self):
        """读取最新帧"""
        try:
            return True, self.frame_buffer.get(timeout=0.1)
        except Empty:
            return False, None

    def get_stats(self):
        """获取统计信息"""
        return {
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'fps': self.fps
        }

    def stop(self):
        """停止捕获"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

def main():
    print("=== Jetson Orin Nano 优化版本 ===")
    
    # 获取模型路径
    possible_model_dirs = [
        "../weights",
        "weights", 
        "../ready/weights",
        "/home/nvidia/weights",  # Jetson常用路径
        "./weights"
    ]
    
    model_dir = None
    for path in possible_model_dirs:
        if os.path.exists(path):
            model_dir = path
            break
    
    if model_dir is None:
        print("未找到模型目录，请检查weights文件夹位置")
        return
    
    # 优先使用TensorRT引擎
    trt_model_path = os.path.join(model_dir, "best_trt.engine")
    pt_model_path = os.path.join(model_dir, "best.pt")
    
    if os.path.exists(trt_model_path):
        model_path = trt_model_path
        print(f"✓ 使用TensorRT引擎: {trt_model_path}")
    elif os.path.exists(pt_model_path):
        model_path = pt_model_path
        print(f"⚠️  使用PyTorch模型: {pt_model_path}")
        print("建议转换为TensorRT引擎以获得更好性能")
    else:
        print("❌ 未找到模型文件")
        return

    # 初始化检测器和处理器
    print("初始化检测器...")
    try:
        # 降低置信度阈值以减少后处理时间
        detector = YOLOTRTDetector(
            model_path=model_path, 
            conf_thres=0.3,  # 提高置信度阈值
            use_trt=model_path.endswith('.engine')
        )
        print("✓ 检测器初始化成功")
        
        processor = OptimizedArrowProcessor()
        print("✓ 处理器初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 视频源设置 - 优先使用摄像头
    video_sources = [
        0,  # 默认摄像头
        "test_video.mp4",  # 测试视频文件
    ]
    
    # 尝试打开视频源
    video_capture = None
    for source in video_sources:
        try:
            temp_cap = FastVideoCapture(source)
            if temp_cap.start():
                video_capture = temp_cap
                print(f"✓ 成功打开视频源: {source}")
                break
            else:
                temp_cap.stop()
        except Exception as e:
            print(f"❌ 打开视频源 {source} 失败: {e}")
    
    if video_capture is None:
        print("❌ 无法打开任何视频源")
        return
    
    # 显示设置
    cv2.namedWindow("Jetson优化检测", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Jetson优化检测", 800, 600)  # 较小的窗口
    
    # 性能统计
    process_times = []
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    
    # 优化参数
    SCALE_PERCENT = 50  # 进一步缩小图像以提高速度
    SKIP_FRAMES = 2     # 跳帧处理，每3帧处理1帧
    MAX_DETECTIONS = 3  # 最多处理3个检测结果
    
    print(f"优化参数: 缩放={SCALE_PERCENT}%, 跳帧={SKIP_FRAMES}, 最大检测数={MAX_DETECTIONS}")
    print("按'q'键退出，按's'键显示统计信息")

    try:
        skip_counter = 0
        
        while True:
            # 读取帧
            ret, frame = video_capture.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # 跳帧策略
            skip_counter += 1
            if skip_counter <= SKIP_FRAMES:
                continue
            skip_counter = 0
            
            # 处理开始计时
            process_start = time.time()
            
            # 大幅缩小图像尺寸
            width = int(frame.shape[1] * SCALE_PERCENT / 100)
            height = int(frame.shape[0] * SCALE_PERCENT / 100)
            frame_small = cv2.resize(frame, (width, height))

            # YOLO检测
            detections = detector.detect(frame_small)
            
            # 限制检测结果数量
            if len(detections) > MAX_DETECTIONS:
                detections = detections[:MAX_DETECTIONS]
            
            # 处理检测结果（简化版）
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det['box'])
                
                try:
                    # 简化的扩展策略
                    expand_ratio = 0.05  # 减少扩展比例
                    width_det = x2 - x1
                    height_det = y2 - y1
                    expand_w = int(width_det * expand_ratio)
                    expand_h = int(height_det * expand_ratio)
                    
                    x1_exp = max(0, x1 - expand_w)
                    y1_exp = max(0, y1 - expand_h)
                    x2_exp = min(frame_small.shape[1], x2 + expand_w)
                    y2_exp = min(frame_small.shape[0], y2 + expand_h)
                    
                    # 裁剪区域
                    crop = frame_small[y1_exp:y2_exp, x1_exp:x2_exp].copy()
                    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                        continue
                    
                    # 快速旋转校正
                    rotated = processor.rotate_arrow_fast(crop)
                    
                    # 快速OCR识别
                    text = processor.ocr_recognize_fast(rotated)
                    
                    # 简化的可视化
                    cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    if text:  # 只有识别到文字才显示
                        cv2.putText(frame_small, text, (x1, y2 + 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # 只显示第一个检测结果的预览
                    if i == 0 and rotated.shape[0] > 0 and rotated.shape[1] > 0:
                        preview_size = 80
                        preview = cv2.resize(rotated, (preview_size, preview_size))
                        if preview_size <= frame_small.shape[0] and preview_size <= frame_small.shape[1]:
                            frame_small[5:5+preview_size, 5:5+preview_size] = preview
                    
                except Exception as e:
                    continue
            
            # 计算处理时间
            process_time = time.time() - process_start
            process_times.append(process_time)
            if len(process_times) > 20:  # 保持最近20帧的数据
                process_times.pop(0)
            
            # 计算FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time
            else:
                fps = 0
            
            # 获取统计信息
            stats = video_capture.get_stats()
            avg_process_time = sum(process_times) / len(process_times) if process_times else 0
            
            # 显示性能信息（简化）
            info_texts = [
                f"FPS: {fps:.1f}",
                f"处理: {avg_process_time*1000:.0f}ms",
                f"丢帧: {stats['dropped_frames']}",
                f"缩放: {SCALE_PERCENT}%"
            ]
            
            for i, text in enumerate(info_texts):
                cv2.putText(frame_small, text, (10, 20 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 显示结果
            cv2.imshow("Jetson优化检测", frame_small)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n=== 性能统计 ===")
                print(f"当前FPS: {fps:.2f}")
                print(f"平均处理时间: {avg_process_time*1000:.0f}ms")
                print(f"总帧数: {stats['total_frames']}")
                print(f"丢帧数: {stats['dropped_frames']}")
                print(f"优化参数: 缩放{SCALE_PERCENT}%, 跳帧{SKIP_FRAMES}")

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
        total_time = time.time() - start_time
        final_stats = video_capture.get_stats() if video_capture else {}
        print(f"\n=== 最终统计 ===")
        print(f"总运行时间: {total_time:.1f}s")
        print(f"总帧数: {final_stats.get('total_frames', 0)}")
        if process_times:
            print(f"平均处理时间: {sum(process_times)/len(process_times)*1000:.0f}ms")
        print(f"优化效果: 从6000ms优化到{sum(process_times)/len(process_times)*1000:.0f}ms" if process_times else "")

if __name__ == "__main__":
    main() 