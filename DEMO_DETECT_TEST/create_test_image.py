#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试用的箭头图像
"""

import cv2
import numpy as np

def create_test_arrow_image():
    """创建一个测试用的箭头图像"""
    
    # 创建一个白色背景
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    # 定义箭头的点（指向右下方）
    arrow_points = np.array([
        [100, 50],   # 箭头尖端
        [70, 70],    # 左上
        [85, 70],    # 内左上
        [85, 130],   # 内左下
        [70, 130],   # 左下
        [100, 150],  # 底部尖端
        [130, 130],  # 右下
        [115, 130],  # 内右下
        [115, 70],   # 内右上
        [130, 70]    # 右上
    ], np.int32)
    
    # 填充箭头为粉色
    cv2.fillPoly(img, [arrow_points], (255, 100, 150))  # 粉色 BGR
    
    # 在箭头上添加数字
    cv2.putText(img, '95', (90, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 保存图像
    output_path = "DEMO_DETECT_TEST/user_arrow.jpg"
    cv2.imwrite(output_path, img)
    
    print(f"✅ 测试箭头图像已创建: {output_path}")
    return output_path

if __name__ == "__main__":
    create_test_arrow_image() 