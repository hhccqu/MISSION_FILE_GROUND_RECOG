#!/usr/bin/env python3
# inference4_enhanced.py
# 基于inference4.py，添加GPS通信、数据记录和存储功能
# 适配Jetson Orin Nano平台

import cv2
import numpy as np
import time
import os
import json
import threading
import queue
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import sqlite3
from pathlib import Path

# GPS和MAVLink通信
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    print("警告: pymavlink未安装，GPS功能将被禁用")
    MAVLINK_AVAILABLE = False

# YOLO和OCR
from yolo_trt_utils import YOLOTRTDetector
import easyocr

@dataclass
class GPSData:
    """GPS数据结构"""
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    heading: float
    speed: float
    fix_type: int
    satellites_visible: int

@dataclass
class DetectionRecord:
    """检测记录数据结构"""
    detection_id: str
    timestamp: float
    datetime_str: str
    # 目标信息
    pixel_x: int
    pixel_y: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    # OCR信息
    ocr_text: str
    ocr_confidence: float
    # GPS信息
    gps_data: Optional[GPSData]
    # 图像信息
    image_width: int
    image_height: int
    crop_image_path: str  # 保存的裁剪图像路径

class GPSReceiver:
    """GPS数据接收器 - 与Pixhawk通信"""
    
    def __init__(self, connection_string: str = "/dev/ttyACM0", baud_rate: int = 57600):
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.master = None
        self.latest_gps = None
        self.running = False
        self.gps_queue = queue.Queue(maxsize=10)
        
    def connect(self) -> bool:
        """连接到飞控"""
        if not MAVLINK_AVAILABLE:
            print("MAVLink不可用，GPS功能禁用")
            return False
            
        try:
            print(f"连接飞控: {self.connection_string}:{self.baud_rate}")
            self.master = mavutil.mavlink_connection(
                self.connection_string, 
                baud=self.baud_rate,
                timeout=3
            )
            
            # 等待心跳
            print("等待飞控心跳...")
            self.master.wait_heartbeat(timeout=10)
            print("飞控连接成功!")
            
            # 请求GPS数据流
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                1,  # 1Hz
                1   # 启用
            )
            
            return True
            
        except Exception as e:
            print(f"连接飞控失败: {e}")
            return False
    
    def start_receiving(self):
        """启动GPS数据接收线程"""
        if not self.master:
            return False
            
        self.running = True
        self.gps_thread = threading.Thread(target=self._gps_receiver_loop, daemon=True)
        self.gps_thread.start()
        return True
    
    def _gps_receiver_loop(self):
        """GPS数据接收循环"""
        while self.running:
            try:
                msg = self.master.recv_match(
                    type=['GLOBAL_POSITION_INT', 'GPS_RAW_INT'], 
                    blocking=True, 
                    timeout=1
                )
                
                if msg is None:
                    continue
                    
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    gps_data = GPSData(
                        timestamp=time.time(),
                        latitude=msg.lat / 1e7,
                        longitude=msg.lon / 1e7,
                        altitude=msg.alt / 1000.0,
                        heading=msg.hdg / 100.0 if msg.hdg != 65535 else 0.0,
                        speed=np.sqrt(msg.vx**2 + msg.vy**2) / 100.0,
                        fix_type=3,  # 假设3D定位
                        satellites_visible=0  # 需要从GPS_RAW_INT获取
                    )
                    
                    self.latest_gps = gps_data
                    
                    # 更新队列
                    try:
                        self.gps_queue.put_nowait(gps_data)
                    except queue.Full:
                        try:
                            self.gps_queue.get_nowait()  # 移除旧数据
                            self.gps_queue.put_nowait(gps_data)
                        except queue.Empty:
                            pass
                            
                elif msg.get_type() == 'GPS_RAW_INT':
                    if self.latest_gps:
                        self.latest_gps.satellites_visible = msg.satellites_visible
                        self.latest_gps.fix_type = msg.fix_type
                        
            except Exception as e:
                print(f"GPS接收错误: {e}")
                time.sleep(0.1)
    
    def get_latest_gps(self) -> Optional[GPSData]:
        """获取最新GPS数据"""
        return self.latest_gps
    
    def stop(self):
        """停止GPS接收"""
        self.running = False
        if hasattr(self, 'gps_thread'):
            self.gps_thread.join(timeout=2)

class DataRecorder:
    """数据记录器"""
    
    def __init__(self, data_dir: str = "/home/lyc/detection_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.images_dir = self.data_dir / "images"
        self.database_dir = self.data_dir / "database"
        self.logs_dir = self.data_dir / "logs"
        
        for dir_path in [self.images_dir, self.database_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 初始化数据库
        self.db_path = self.database_dir / "detections.db"
        self._init_database()
        
        # 计数器
        self.detection_counter = 0
        
    def _init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id TEXT UNIQUE,
                timestamp REAL,
                datetime_str TEXT,
                pixel_x INTEGER,
                pixel_y INTEGER,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                confidence REAL,
                ocr_text TEXT,
                ocr_confidence REAL,
                gps_latitude REAL,
                gps_longitude REAL,
                gps_altitude REAL,
                gps_heading REAL,
                gps_speed REAL,
                gps_fix_type INTEGER,
                gps_satellites INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                crop_image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_detection(self, record: DetectionRecord, crop_image: np.ndarray) -> str:
        """保存检测记录"""
        # 保存裁剪图像
        image_filename = f"detection_{record.detection_id}.jpg"
        image_path = self.images_dir / image_filename
        cv2.imwrite(str(image_path), crop_image)
        record.crop_image_path = str(image_path)
        
        # 保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO detections (
                    detection_id, timestamp, datetime_str,
                    pixel_x, pixel_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    confidence, ocr_text, ocr_confidence,
                    gps_latitude, gps_longitude, gps_altitude, 
                    gps_heading, gps_speed, gps_fix_type, gps_satellites,
                    image_width, image_height, crop_image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.detection_id, record.timestamp, record.datetime_str,
                record.pixel_x, record.pixel_y, 
                record.bbox[0], record.bbox[1], record.bbox[2], record.bbox[3],
                record.confidence, record.ocr_text, record.ocr_confidence,
                record.gps_data.latitude if record.gps_data else None,
                record.gps_data.longitude if record.gps_data else None,
                record.gps_data.altitude if record.gps_data else None,
                record.gps_data.heading if record.gps_data else None,
                record.gps_data.speed if record.gps_data else None,
                record.gps_data.fix_type if record.gps_data else None,
                record.gps_data.satellites_visible if record.gps_data else None,
                record.image_width, record.image_height, record.crop_image_path
            ))
            
            conn.commit()
            print(f"保存检测记录: {record.detection_id}")
            
        except sqlite3.IntegrityError:
            print(f"检测记录已存在: {record.detection_id}")
        except Exception as e:
            print(f"保存检测记录失败: {e}")
        finally:
            conn.close()
        
        # 同时保存JSON格式备份
        json_path = self.logs_dir / f"detection_{record.detection_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(record), f, indent=2, ensure_ascii=False)
        
        return record.detection_id
    
    def generate_detection_id(self) -> str:
        """生成唯一的检测ID"""
        self.detection_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"DET_{timestamp}_{self.detection_counter:04d}"

class ArrowProcessor:
    def __init__(self):
        # 初始化OCR（全局单例）
        print("初始化EasyOCR (CPU模式)")
        self.reader = easyocr.Reader(['en'], gpu=False, download_enabled=False)
        
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
        """执行OCR识别，返回文本和置信度"""
        # 预处理增强对比度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 执行OCR
        results = self.reader.readtext(enhanced, detail=1)  # 获取详细信息
        
        if results:
            # 合并所有文本和计算平均置信度
            texts = []
            confidences = []
            for (bbox, text, conf) in results:
                texts.append(text.upper())
                confidences.append(conf)
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return combined_text, avg_confidence
        else:
            return "", 0.0

def main():
    print("启动增强版检测系统...")
    
    # 获取模型路径
    model_dir = "/home/lyc/CQU_Ground_ReconnaissanceStrike/weights"
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
        detector = YOLOTRTDetector(model_path=model_path, conf_thres=0.25, use_trt=True)
        processor = ArrowProcessor()
        recorder = DataRecorder()
        
        # 初始化GPS接收器
        gps_receiver = GPSReceiver(connection_string="/dev/ttyACM0", baud_rate=57600)
        gps_connected = gps_receiver.connect()
        
        if gps_connected:
            gps_receiver.start_receiving()
            print("GPS接收器启动成功")
        else:
            print("GPS接收器启动失败，将在无GPS模式下运行")
        
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 视频输入设置
    video_path = "/home/lyc/CQU_Ground_ReconnaissanceStrike/video1.mp4"
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"无法打开视频源: {video_path}")
    
    # 显示设置
    cv2.namedWindow("Enhanced Detection System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Enhanced Detection System", 1280, 720)
    
    # 性能统计
    frame_count = 0
    fps_avg = 0
    start_time = time.time()
    detection_count = 0
    
    print("系统就绪! 按'q'键退出, 按's'键保存当前检测")

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
            cv2.imshow("Enhanced Detection System", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
    
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        
        if gps_connected:
            gps_receiver.stop()
        
        print(f"\n系统统计:")
        print(f"总检测数: {detection_count}")
        print(f"平均FPS: {fps_avg:.2f}")
        print(f"数据保存目录: {recorder.data_dir}")

if __name__ == "__main__":
    main() 