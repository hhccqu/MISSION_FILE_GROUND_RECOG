import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import json
import time

@dataclass
class FlightData:
    """飞行数据结构"""
    timestamp: float
    latitude: float  # 纬度（度）
    longitude: float  # 经度（度）
    altitude: float  # 高度（米）
    pitch: float  # 俯仰角（度）
    roll: float  # 横滚角（度）
    yaw: float  # 偏航角（度）
    ground_speed: float  # 地面速度（m/s）
    heading: float  # 航向角（度）

@dataclass
class TargetInfo:
    """目标信息结构"""
    target_id: str
    detected_number: str
    pixel_x: int
    pixel_y: int
    confidence: float
    latitude: float
    longitude: float
    flight_data: FlightData
    timestamp: float

class GPSSimulator:
    """GPS模拟器 - 模拟直线飞行"""
    
    def __init__(self, start_lat=30.6586, start_lon=104.0647, altitude=500.0, speed=30.0, heading=90.0):
        """
        初始化GPS模拟器
        
        参数:
            start_lat: 起始纬度（默认成都）
            start_lon: 起始经度
            altitude: 飞行高度（米）
            speed: 飞行速度（m/s）
            heading: 航向角（度，0=北，90=东）
        """
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.altitude = altitude
        self.speed = speed
        self.heading = heading
        self.start_time = time.time()
        
        # 模拟飞行参数
        self.pitch_base = -10.0  # 基础俯仰角（向下观察）
        self.roll_amplitude = 2.0  # 横滚角振幅
        self.pitch_amplitude = 1.0  # 俯仰角振幅
        
    def get_current_position(self) -> FlightData:
        """获取当前飞行数据"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 计算距离（米）
        distance = self.speed * elapsed_time
        
        # 将距离转换为经纬度偏移
        lat_offset, lon_offset = self._meters_to_degrees(distance, self.heading, self.start_lat)
        
        # 当前位置
        current_lat = self.start_lat + lat_offset
        current_lon = self.start_lon + lon_offset
        
        # 模拟姿态变化（轻微振荡）
        pitch = self.pitch_base + self.pitch_amplitude * math.sin(elapsed_time * 0.5)
        roll = self.roll_amplitude * math.sin(elapsed_time * 0.3)
        yaw = self.heading + 1.0 * math.sin(elapsed_time * 0.2)  # 轻微偏航
        
        return FlightData(
            timestamp=current_time,
            latitude=current_lat,
            longitude=current_lon,
            altitude=self.altitude,
            pitch=pitch,
            roll=roll,
            yaw=yaw,
            ground_speed=self.speed,
            heading=self.heading
        )
    
    def _meters_to_degrees(self, distance_meters: float, heading_degrees: float, latitude: float) -> Tuple[float, float]:
        """将米距离转换为经纬度偏移"""
        # 地球半径（米）
        earth_radius = 6378137.0
        
        # 将航向角转换为弧度
        heading_rad = math.radians(heading_degrees)
        
        # 计算纬度偏移
        lat_offset = (distance_meters * math.cos(heading_rad)) / earth_radius * 180.0 / math.pi
        
        # 计算经度偏移（考虑纬度影响）
        lon_offset = (distance_meters * math.sin(heading_rad)) / (earth_radius * math.cos(math.radians(latitude))) * 180.0 / math.pi
        
        return lat_offset, lon_offset

class TargetGeoCalculator:
    """目标地理坐标计算器"""
    
    def __init__(self, camera_fov_h=60.0, camera_fov_v=45.0, image_width=1920, image_height=1080):
        """
        初始化目标地理坐标计算器
        
        参数:
            camera_fov_h: 相机水平视场角（度）
            camera_fov_v: 相机垂直视场角（度）
            image_width: 图像宽度（像素）
            image_height: 图像高度（像素）
        """
        self.camera_fov_h = math.radians(camera_fov_h)
        self.camera_fov_v = math.radians(camera_fov_v)
        self.image_width = image_width
        self.image_height = image_height
        
    def calculate_target_position(self, pixel_x: int, pixel_y: int, flight_data: FlightData) -> Tuple[float, float]:
        """
        根据像素位置和飞行数据计算目标GPS坐标
        
        参数:
            pixel_x: 目标在图像中的x坐标
            pixel_y: 目标在图像中的y坐标
            flight_data: 当前飞行数据
            
        返回:
            (target_lat, target_lon): 目标的GPS坐标
        """
        # 将像素坐标转换为相机坐标系中的角度
        # 图像中心为原点，x向右为正，y向下为正
        center_x = self.image_width / 2
        center_y = self.image_height / 2
        
        # 计算相对于图像中心的偏移（归一化）
        norm_x = (pixel_x - center_x) / center_x
        norm_y = (pixel_y - center_y) / center_y
        
        # 转换为相机坐标系中的角度
        angle_x = norm_x * (self.camera_fov_h / 2)  # 水平角度
        angle_y = norm_y * (self.camera_fov_v / 2)  # 垂直角度
        
        # 考虑飞机姿态，计算相对于地面的角度
        # 俯仰角影响垂直方向，偏航角影响水平方向
        pitch_rad = math.radians(flight_data.pitch)
        yaw_rad = math.radians(flight_data.yaw)
        roll_rad = math.radians(flight_data.roll)
        
        # 计算目标相对于飞机的地面投影距离
        # 考虑俯仰角和相机角度
        ground_angle_y = pitch_rad + angle_y
        
        # 如果角度太小（接近水平），则无法准确定位
        if abs(ground_angle_y) < math.radians(5):
            ground_angle_y = math.radians(5) if ground_angle_y >= 0 else math.radians(-5)
        
        # 计算地面距离
        ground_distance = flight_data.altitude / math.tan(abs(ground_angle_y))
        
        # 计算水平偏移
        horizontal_offset = ground_distance * math.tan(angle_x)
        
        # 考虑横滚角的影响（简化处理）
        corrected_distance = ground_distance * math.cos(roll_rad)
        corrected_offset = horizontal_offset * math.cos(roll_rad)
        
        # 计算目标相对于飞机的北东坐标
        # 考虑飞机航向角
        north_offset = corrected_distance * math.cos(yaw_rad) - corrected_offset * math.sin(yaw_rad)
        east_offset = corrected_distance * math.sin(yaw_rad) + corrected_offset * math.cos(yaw_rad)
        
        # 转换为GPS坐标
        target_lat, target_lon = self._offset_to_gps(
            flight_data.latitude, flight_data.longitude, north_offset, east_offset
        )
        
        return target_lat, target_lon
    
    def _offset_to_gps(self, base_lat: float, base_lon: float, north_meters: float, east_meters: float) -> Tuple[float, float]:
        """将北东偏移转换为GPS坐标"""
        # 地球半径（米）
        earth_radius = 6378137.0
        
        # 纬度偏移
        lat_offset = north_meters / earth_radius * 180.0 / math.pi
        
        # 经度偏移（考虑纬度影响）
        lon_offset = east_meters / (earth_radius * math.cos(math.radians(base_lat))) * 180.0 / math.pi
        
        return base_lat + lat_offset, base_lon + lon_offset

class OCRNumberExtractor:
    """OCR数字提取器"""
    
    @staticmethod
    def extract_two_digit_numbers(ocr_text: str) -> list:
        """
        从OCR文本中提取二位数
        
        参数:
            ocr_text: OCR识别的文本
            
        返回:
            提取到的二位数列表
        """
        import re
        
        # 清理文本，移除特殊字符
        cleaned_text = re.sub(r'[^\d\s]', '', ocr_text)
        
        # 查找所有数字
        numbers = re.findall(r'\d+', cleaned_text)
        
        # 筛选二位数
        two_digit_numbers = []
        for num in numbers:
            if len(num) == 2:
                two_digit_numbers.append(num)
            elif len(num) > 2:
                # 如果是多位数，尝试分割为二位数
                for i in range(0, len(num) - 1, 2):
                    if i + 1 < len(num):
                        two_digit = num[i:i+2]
                        two_digit_numbers.append(two_digit)
        
        return two_digit_numbers

class TargetDataManager:
    """目标数据管理器"""
    
    def __init__(self, save_file="target_data.json"):
        """
        初始化目标数据管理器
        
        参数:
            save_file: 保存文件路径
        """
        self.save_file = save_file
        self.targets = []
        
    def add_target(self, target_info: TargetInfo):
        """添加目标信息"""
        self.targets.append(target_info)
        
    def save_to_file(self):
        """保存目标数据到文件"""
        data = []
        for target in self.targets:
            data.append({
                "target_id": target.target_id,
                "detected_number": target.detected_number,
                "pixel_position": {
                    "x": target.pixel_x,
                    "y": target.pixel_y
                },
                "confidence": target.confidence,
                "gps_position": {
                    "latitude": target.latitude,
                    "longitude": target.longitude
                },
                "flight_data": {
                    "timestamp": target.flight_data.timestamp,
                    "latitude": target.flight_data.latitude,
                    "longitude": target.flight_data.longitude,
                    "altitude": target.flight_data.altitude,
                    "pitch": target.flight_data.pitch,
                    "roll": target.flight_data.roll,
                    "yaw": target.flight_data.yaw,
                    "ground_speed": target.flight_data.ground_speed,
                    "heading": target.flight_data.heading
                },
                "detection_timestamp": target.timestamp
            })
        
        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_targets_count(self) -> int:
        """获取目标数量"""
        return len(self.targets)
    
    def clear_targets(self):
        """清空目标数据"""
        self.targets.clear() 