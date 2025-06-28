#!/usr/bin/env python3
# utils/visualization.py
# 可视化工具函数

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List

def draw_detection(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    text: str = "",
    confidence: float = 0.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制检测框和文本
    
    参数:
        frame: 原始图像
        bbox: 边界框坐标 (x1, y1, x2, y2)
        text: 显示文本
        confidence: 置信度
        color: 边界框颜色
        text_color: 文本颜色
        thickness: 线条粗细
        
    返回:
        绘制后的图像
    """
    x1, y1, x2, y2 = bbox
    
    # 绘制边界框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # 准备显示文本
    if text and confidence > 0:
        display_text = f"{text} ({confidence:.2f})"
    elif text:
        display_text = text
    elif confidence > 0:
        display_text = f"conf: {confidence:.2f}"
    else:
        display_text = ""
    
    # 绘制文本背景
    if display_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_size = cv2.getTextSize(display_text, font, font_scale, font_thickness)[0]
        
        # 文本框位置
        text_x = x1
        text_y = y2 + 25
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (text_x, text_y - text_size[1] - 5), 
            (text_x + text_size[0] + 5, text_y + 5), 
            (0, 0, 0), 
            -1
        )
        # 应用透明度
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 绘制文本
        cv2.putText(
            frame, 
            display_text, 
            (text_x + 2, text_y - 2), 
            font, 
            font_scale, 
            text_color, 
            font_thickness
        )
    
    return frame

def draw_gps_info(
    frame: np.ndarray,
    latitude: float,
    longitude: float,
    altitude: Optional[float] = None,
    heading: Optional[float] = None,
    satellites: Optional[int] = None,
    position: str = "bottom",
    color: Tuple[int, int, int] = (255, 255, 0)
) -> np.ndarray:
    """
    在图像上绘制GPS信息
    
    参数:
        frame: 原始图像
        latitude: 纬度
        longitude: 经度
        altitude: 高度（可选）
        heading: 航向（可选）
        satellites: 卫星数量（可选）
        position: 显示位置 ("top", "bottom")
        color: 文本颜色
        
    返回:
        绘制后的图像
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    line_height = 25
    
    # 准备显示文本
    texts = []
    texts.append(f"位置: {latitude:.6f}, {longitude:.6f}")
    
    if altitude is not None:
        texts.append(f"高度: {altitude:.1f}m")
    
    if heading is not None and satellites is not None:
        texts.append(f"航向: {heading:.1f}° | 卫星: {satellites}")
    elif heading is not None:
        texts.append(f"航向: {heading:.1f}°")
    elif satellites is not None:
        texts.append(f"卫星: {satellites}")
    
    # 确定绘制位置
    if position.lower() == "top":
        y_start = 30
        y_increment = line_height
    else:  # bottom
        y_start = h - len(texts) * line_height
        y_increment = line_height
    
    # 绘制半透明背景
    overlay = frame.copy()
    bg_height = len(texts) * line_height + 10
    cv2.rectangle(
        overlay, 
        (5, y_start - 20), 
        (w - 5, y_start + bg_height), 
        (0, 0, 0), 
        -1
    )
    # 应用透明度
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 绘制文本
    for i, text in enumerate(texts):
        y = y_start + i * y_increment
        cv2.putText(frame, text, (10, y), font, font_scale, color, font_thickness)
    
    return frame

def draw_status_bar(
    frame: np.ndarray,
    status_items: Dict[str, str],
    position: str = "top",
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.7
) -> np.ndarray:
    """
    在图像上绘制状态栏
    
    参数:
        frame: 原始图像
        status_items: 状态项字典 {"名称": "值"}
        position: 显示位置 ("top", "bottom")
        bg_color: 背景颜色
        text_color: 文本颜色
        alpha: 背景透明度
        
    返回:
        绘制后的图像
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    bar_height = 40
    
    # 确定绘制位置
    if position.lower() == "top":
        y_bar = 0
    else:  # bottom
        y_bar = h - bar_height
    
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_bar), (w, y_bar + bar_height), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 计算每个状态项的宽度
    num_items = len(status_items)
    item_width = w // num_items
    
    # 绘制状态项
    for i, (key, value) in enumerate(status_items.items()):
        x = i * item_width + 10
        y = y_bar + bar_height // 2 + 5
        text = f"{key}: {value}"
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness)
    
    return frame