#!/usr/bin/env python3
# utils/conversion.py
# 坐标转换工具函数

import numpy as np
from typing import Tuple, Optional

def pixel_to_geo(
    pixel_x: int,
    pixel_y: int,
    image_width: int,
    image_height: int,
    ref_lat: float,
    ref_lon: float,
    heading: float = 0.0,
    fov_h: float = 60.0,
    fov_v: float = 45.0,
    altitude: float = 100.0
) -> Tuple[float, float]:
    """
    将图像像素坐标转换为地理坐标
    
    参数:
        pixel_x: 像素X坐标
        pixel_y: 像素Y坐标
        image_width: 图像宽度
        image_height: 图像高度
        ref_lat: 参考纬度（相机位置）
        ref_lon: 参考经度（相机位置）
        heading: 相机航向角（度）
        fov_h: 水平视场角（度）
        fov_v: 垂直视场角（度）
        altitude: 相机高度（米）
        
    返回:
        (纬度, 经度) 元组
    """
    # 将像素坐标归一化到[-1, 1]范围
    norm_x = (pixel_x / image_width) * 2 - 1
    norm_y = (pixel_y / image_height) * 2 - 1
    
    # 计算视角偏移（弧度）
    angle_h = np.radians(norm_x * (fov_h / 2))
    angle_v = np.radians(norm_y * (fov_v / 2))
    
    # 计算地面投影距离
    distance = altitude * np.tan(angle_v)
    
    # 计算地面投影点相对于相机的偏移（米）
    heading_rad = np.radians(heading)
    east_offset = distance * np.sin(heading_rad + angle_h)
    north_offset = distance * np.cos(heading_rad + angle_h)
    
    # 转换为经纬度偏移（近似计算）
    # 地球半径约6371000米
    earth_radius = 6371000.0
    lat_offset = np.degrees(north_offset / earth_radius)
    lon_offset = np.degrees(east_offset / (earth_radius * np.cos(np.radians(ref_lat))))
    
    # 计算目标经纬度
    target_lat = ref_lat + lat_offset
    target_lon = ref_lon + lon_offset
    
    return target_lat, target_lon

def geo_to_pixel(
    lat: float,
    lon: float,
    image_width: int,
    image_height: int,
    ref_lat: float,
    ref_lon: float,
    heading: float = 0.0,
    fov_h: float = 60.0,
    fov_v: float = 45.0,
    altitude: float = 100.0
) -> Optional[Tuple[int, int]]:
    """
    将地理坐标转换为图像像素坐标
    
    参数:
        lat: 目标纬度
        lon: 目标经度
        image_width: 图像宽度
        image_height: 图像高度
        ref_lat: 参考纬度（相机位置）
        ref_lon: 参考经度（相机位置）
        heading: 相机航向角（度）
        fov_h: 水平视场角（度）
        fov_v: 垂直视场角（度）
        altitude: 相机高度（米）
        
    返回:
        (像素X, 像素Y) 元组，如果点不在视野内则返回None
    """
    # 地球半径约6371000米
    earth_radius = 6371000.0
    
    # 计算经纬度偏移
    lat_offset = lat - ref_lat
    lon_offset = lon - ref_lon
    
    # 转换为米偏移
    north_offset = np.radians(lat_offset) * earth_radius
    east_offset = np.radians(lon_offset) * earth_radius * np.cos(np.radians(ref_lat))
    
    # 计算地面距离
    distance = np.sqrt(north_offset**2 + east_offset**2)
    
    # 如果距离太远，可能不在视野内
    max_visible_distance = altitude * np.tan(np.radians(max(fov_h, fov_v)))
    if distance > max_visible_distance:
        return None
    
    # 计算相对于相机航向的角度
    heading_rad = np.radians(heading)
    target_angle = np.arctan2(east_offset, north_offset) - heading_rad
    
    # 归一化角度到[-π, π]
    target_angle = ((target_angle + np.pi) % (2 * np.pi)) - np.pi
    
    # 计算视角
    angle_h = target_angle
    angle_v = np.arctan(distance / altitude)
    
    # 检查是否在视野范围内
    if abs(np.degrees(angle_h)) > fov_h/2 or abs(np.degrees(angle_v)) > fov_v/2:
        return None
    
    # 转换为归一化坐标 [-1, 1]
    norm_x = np.degrees(angle_h) / (fov_h / 2)
    norm_y = np.degrees(angle_v) / (fov_v / 2)
    
    # 转换为像素坐标
    pixel_x = int((norm_x + 1) / 2 * image_width)
    pixel_y = int((norm_y + 1) / 2 * image_height)
    
    # 确保坐标在图像范围内
    pixel_x = max(0, min(image_width - 1, pixel_x))
    pixel_y = max(0, min(image_height - 1, pixel_y))
    
    return pixel_x, pixel_y 