#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机对地打击任务系统
整合目标检测、OCR识别、GPS定位和数据保存
"""

import cv2
import numpy as np
import time
import os
import threading
from queue import Queue, Empty
import uuid
from datetime import datetime

# 导入自定义模块
from yolo_trt_utils import YOLOTRTDetector
from target_geo_calculator import (
    GPSSimulator, TargetGeoCalculator, OCRNumberExtractor, 
    TargetDataManager, FlightData, TargetInfo
)
import easyocr

class StrikeMissionSystem:
    """无人机对地打击任务系统"""
    
    def __init__(self, config=None):
        """
        初始化打击任务系统
        
        参数:
            config: 配置字典
        """
        # 默认配置
        self.config = {
            'model_path': None,
            'conf_threshold': 0.25,
            'camera_fov_h': 60.0,
            'camera_fov_v': 45.0,
            'start_lat': 30.6586,  # 成都
            'start_lon': 104.0647,
            'altitude': 500.0,
            'speed': 30.0,
            'heading': 90.0,
            'save_file': 'strike_targets.json',
            'min_confidence': 0.5,
            'ocr_interval': 5,
            'max_targets_per_frame': 5
        }
        
        if config:
            self.config.update(config)
        
        # 初始化组件
        self.detector = None
        self.ocr_reader = None
        self.gps_simulator = None
        self.geo_calculator = None
        self.data_manager = None
        self.number_extractor = OCRNumberExtractor()
        
        # 运行状态
        self.running = False
        self.frame_count = 0
        self.detection_count = 0
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'valid_targets': 0,
            'ocr_success': 0,
            'start_time': None
        }
        
    def initialize(self):
        """初始化所有组件"""
        print("🚀 初始化无人机对地打击任务系统...")
        
        # 1. 初始化YOLO检测器
        print("📡 初始化目标检测器...")
        model_path = self._find_model()
        if not model_path:
            raise RuntimeError("未找到模型文件")
        
        self.detector = YOLOTRTDetector(
            model_path=model_path, 
            conf_thres=self.config['conf_threshold']
        )
        
        # 2. 初始化OCR
        print("🔤 初始化OCR识别器...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        
        # 3. 初始化GPS模拟器
        print("🛰️ 初始化GPS模拟器...")
        self.gps_simulator = GPSSimulator(
            start_lat=self.config['start_lat'],
            start_lon=self.config['start_lon'],
            altitude=self.config['altitude'],
            speed=self.config['speed'],
            heading=self.config['heading']
        )
        
        # 4. 初始化地理坐标计算器
        print("📍 初始化地理坐标计算器...")
        self.geo_calculator = TargetGeoCalculator(
            camera_fov_h=self.config['camera_fov_h'],
            camera_fov_v=self.config['camera_fov_v']
        )
        
        # 5. 初始化数据管理器
        print("💾 初始化数据管理器...")
        self.data_manager = TargetDataManager(self.config['save_file'])
        
        print("✅ 系统初始化完成！")
        
    def _find_model(self):
        """查找模型文件"""
        possible_model_paths = [
            "../weights/best1.pt",
            "weights/best1.pt", 
            "../ready/weights/best1.pt",
            "D:/AirmodelingTeam/CQU_Ground_Recog_Strile_YoloOcr/weights/best1.pt"
        ]
        
        for path in possible_model_paths:
            if os.path.exists(path):
                return path
        return None
        
    def process_frame(self, frame):
        """
        处理单帧图像
        
        参数:
            frame: 输入图像
            
        返回:
            processed_frame: 处理后的图像
            target_count: 检测到的目标数量
        """
        self.frame_count += 1
        current_time = time.time()
        
        # 获取当前飞行数据
        flight_data = self.gps_simulator.get_current_position()
        
        # YOLO目标检测
        detections = self.detector.detect(frame)
        self.stats['total_detections'] += len(detections)
        
        # 更新图像尺寸（动态获取）
        height, width = frame.shape[:2]
        self.geo_calculator.image_height = height
        self.geo_calculator.image_width = width
        
        # 处理检测结果
        valid_targets = 0
        processed_frame = frame.copy()
        
        # 限制处理的目标数量
        max_targets = min(len(detections), self.config['max_targets_per_frame'])
        
        for i, det in enumerate(detections[:max_targets]):
            if det['confidence'] < self.config['min_confidence']:
                continue
                
            x1, y1, x2, y2 = map(int, det['box'])
            
            try:
                # 扩展检测框
                expand_ratio = 0.1
                w, h = x2 - x1, y2 - y1
                x1_exp = max(0, x1 - int(w * expand_ratio))
                y1_exp = max(0, y1 - int(h * expand_ratio))
                x2_exp = min(width, x2 + int(w * expand_ratio))
                y2_exp = min(height, y2 + int(h * expand_ratio))
                
                # 裁剪目标区域
                crop = frame[y1_exp:y2_exp, x1_exp:x2_exp].copy()
                if crop.size == 0:
                    continue
                
                # 旋转校正（使用原有的ArrowProcessor逻辑）
                rotated = self._rotate_arrow(crop)
                
                # OCR识别（每隔几帧进行一次）
                ocr_text = ""
                if self.frame_count % self.config['ocr_interval'] == 0:
                    ocr_text = self._perform_ocr(rotated)
                    if ocr_text:
                        self.stats['ocr_success'] += 1
                
                # 提取二位数
                numbers = self.number_extractor.extract_two_digit_numbers(ocr_text)
                detected_number = numbers[0] if numbers else "未识别"
                
                # 计算目标中心像素坐标
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # 计算目标GPS坐标
                target_lat, target_lon = self.geo_calculator.calculate_target_position(
                    center_x, center_y, flight_data
                )
                
                # 创建目标信息
                target_info = TargetInfo(
                    target_id=f"T{self.detection_count:04d}",
                    detected_number=detected_number,
                    pixel_x=center_x,
                    pixel_y=center_y,
                    confidence=det['confidence'],
                    latitude=target_lat,
                    longitude=target_lon,
                    flight_data=flight_data,
                    timestamp=current_time
                )
                
                # 保存目标数据
                self.data_manager.add_target(target_info)
                self.detection_count += 1
                valid_targets += 1
                
                # 在图像上绘制结果
                self._draw_detection_result(
                    processed_frame, x1, y1, x2, y2, 
                    target_info, rotated
                )
                
            except Exception as e:
                print(f"处理目标 {i} 时出错: {e}")
                continue
        
        # 绘制飞行信息
        self._draw_flight_info(processed_frame, flight_data)
        
        # 绘制统计信息
        self._draw_statistics(processed_frame, valid_targets)
        
        return processed_frame, valid_targets
    
    def _rotate_arrow(self, crop_image):
        """箭头旋转校正（简化版）"""
        try:
            # 转换为HSV进行红色检测
            hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
            
            # 红色范围
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([30, 255, 255])
            lower_red2 = np.array([150, 30, 30])
            upper_red2 = np.array([179, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大轮廓
                max_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(max_contour)
                (_, _), (w, h), angle = rect
                
                # 角度修正
                if w > h:
                    angle += 90
                
                # 执行旋转
                (h, w) = crop_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(crop_image, M, (w, h), borderValue=(255, 255, 255))
                
                return rotated
            
        except Exception as e:
            print(f"旋转校正失败: {e}")
        
        return crop_image
    
    def _perform_ocr(self, image):
        """执行OCR识别"""
        try:
            # 预处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # OCR识别
            results = self.ocr_reader.readtext(enhanced, detail=0)
            return " ".join(results).upper()
            
        except Exception as e:
            print(f"OCR识别失败: {e}")
            return ""
    
    def _draw_detection_result(self, frame, x1, y1, x2, y2, target_info, rotated_crop):
        """在图像上绘制检测结果"""
        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制目标信息
        info_text = f"ID:{target_info.target_id} 数字:{target_info.detected_number}"
        cv2.putText(frame, info_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制GPS坐标
        gps_text = f"GPS:{target_info.latitude:.6f},{target_info.longitude:.6f}"
        cv2.putText(frame, gps_text, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 绘制置信度
        conf_text = f"Conf:{target_info.confidence:.2f}"
        cv2.putText(frame, conf_text, (x1, y2 + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    
    def _draw_flight_info(self, frame, flight_data):
        """绘制飞行信息"""
        info_lines = [
            f"GPS: {flight_data.latitude:.6f}, {flight_data.longitude:.6f}",
            f"高度: {flight_data.altitude:.1f}m",
            f"姿态: P{flight_data.pitch:.1f}° R{flight_data.roll:.1f}° Y{flight_data.yaw:.1f}°",
            f"速度: {flight_data.ground_speed:.1f}m/s 航向: {flight_data.heading:.1f}°"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_statistics(self, frame, current_targets):
        """绘制统计信息"""
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            fps = self.frame_count / elapsed if elapsed > 0 else 0
        else:
            fps = 0
        
        stats_lines = [
            f"帧数: {self.frame_count}",
            f"FPS: {fps:.1f}",
            f"当前目标: {current_targets}",
            f"总检测: {self.stats['total_detections']}",
            f"有效目标: {self.data_manager.get_targets_count()}",
            f"OCR成功: {self.stats['ocr_success']}"
        ]
        
        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (frame.shape[1] - 200, 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def run_video_mission(self, video_source=0):
        """运行视频任务"""
        print(f"🎯 开始执行打击任务，视频源: {video_source}")
        
        # 打开视频源
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {video_source}")
        
        # 设置窗口
        cv2.namedWindow("无人机对地打击任务", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("无人机对地打击任务", 1280, 720)
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        print("📋 任务控制:")
        print("  'q' - 退出任务")
        print("  's' - 保存当前数据")
        print("  'r' - 重置统计")
        print("  'c' - 清空目标数据")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("视频读取结束或失败")
                    break
                
                # 处理帧
                processed_frame, target_count = self.process_frame(frame)
                
                # 显示结果
                cv2.imshow("无人机对地打击任务", processed_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_data()
                elif key == ord('r'):
                    self._reset_stats()
                elif key == ord('c'):
                    self._clear_data()
                
                # 自动保存（每100个目标）
                if self.data_manager.get_targets_count() % 100 == 0 and self.data_manager.get_targets_count() > 0:
                    self._save_data()
                    
        except KeyboardInterrupt:
            print("\n收到中断信号")
        finally:
            self._cleanup(cap)
    
    def _save_data(self):
        """保存数据"""
        try:
            self.data_manager.save_to_file()
            print(f"✅ 数据已保存到 {self.config['save_file']}")
            print(f"📊 总计 {self.data_manager.get_targets_count()} 个目标")
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
    
    def _reset_stats(self):
        """重置统计"""
        self.stats = {
            'total_detections': 0,
            'valid_targets': 0,
            'ocr_success': 0,
            'start_time': time.time()
        }
        self.frame_count = 0
        print("📊 统计信息已重置")
    
    def _clear_data(self):
        """清空目标数据"""
        self.data_manager.clear_targets()
        self.detection_count = 0
        print("🗑️ 目标数据已清空")
    
    def _cleanup(self, cap):
        """清理资源"""
        print("\n🔄 正在清理资源...")
        
        # 最终保存
        self._save_data()
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 打印最终统计
        print("\n📈 最终统计:")
        print(f"  处理帧数: {self.frame_count}")
        print(f"  检测总数: {self.stats['total_detections']}")
        print(f"  有效目标: {self.data_manager.get_targets_count()}")
        print(f"  OCR成功: {self.stats['ocr_success']}")
        
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            print(f"  运行时间: {elapsed:.1f}秒")
            print(f"  平均FPS: {self.frame_count / elapsed:.1f}")
        
        print("✅ 任务完成!")

def main():
    """主函数"""
    # 配置参数
    config = {
        'conf_threshold': 0.25,
        'camera_fov_h': 60.0,
        'camera_fov_v': 45.0,
        'start_lat': 30.6586,  # 成都坐标
        'start_lon': 104.0647,
        'altitude': 500.0,
        'speed': 30.0,
        'heading': 90.0,
        'save_file': 'strike_targets.json',
        'min_confidence': 0.5,
        'ocr_interval': 5,
        'max_targets_per_frame': 5
    }
    
    # 视频源选项
    video_sources = [
        "D:/AirmodelingTeam/CQU_Ground_Recog_Strile_YoloOcr/video2.mp4",
        0,  # 摄像头
        1,
    ]
    
    # 创建任务系统
    mission = StrikeMissionSystem(config)
    
    try:
        # 初始化系统
        mission.initialize()
        
        # 尝试打开视频源
        for source in video_sources:
            try:
                mission.run_video_mission(source)
                break
            except Exception as e:
                print(f"视频源 {source} 失败: {e}")
                continue
        else:
            print("❌ 所有视频源都无法打开")
            
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 