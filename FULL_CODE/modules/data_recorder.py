#!/usr/bin/env python3
# modules/data_recorder.py
# 数据记录模块 - 负责检测结果的存储和管理

import os
import json
import sqlite3
import cv2
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Tuple, Optional
import numpy as np

# 导入GPS数据类型
from modules.gps_receiver import GPSData

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
        
        # 由于GPSData对象不能直接序列化为JSON，需要特殊处理
        record_dict = asdict(record)
        if record.gps_data:
            record_dict['gps_data'] = asdict(record.gps_data)
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(record_dict, f, indent=2, ensure_ascii=False)
        
        return record.detection_id
    
    def generate_detection_id(self) -> str:
        """生成唯一的检测ID"""
        self.detection_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"DET_{timestamp}_{self.detection_counter:04d}"
        
    def query_detections(self, limit: int = 10):
        """查询最近的检测记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT detection_id, datetime_str, ocr_text, gps_latitude, gps_longitude
            FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results 