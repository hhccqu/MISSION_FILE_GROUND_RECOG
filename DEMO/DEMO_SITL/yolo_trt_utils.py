#!/usr/bin/env python3
# yolo_trt_utils.py
# TensorRT支持的YOLO检测器（兼容版本）

import os
from ultralytics import YOLO

class YOLOTRTDetector:
    """TensorRT支持的YOLO检测器"""
    
    def __init__(self, model_path="weights/best1.pt", conf_thres=0.25, use_trt=True):
        """
        初始化YOLO检测器
        
        参数:
            model_path: 模型文件路径
            conf_thres: 置信度阈值
            use_trt: 是否使用TensorRT（此版本仅为兼容性）
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.use_trt = use_trt and model_path.endswith('.engine')
        self.using_trt = self.use_trt  # 兼容属性
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        try:
            self.model = YOLO(model_path)
            print(f"成功加载模型: {model_path}")
            if self.use_trt:
                print("使用TensorRT引擎")
            else:
                print("使用PyTorch模型")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def detect(self, frame):
        """
        对单帧图像进行检测
        
        参数:
            frame: 输入图像
            
        返回:
            检测结果列表，格式：
            [
                {
                    'box': (x1, y1, x2, y2),
                    'confidence': 0.9,
                    'class_id': 0
                },
                ...
            ]
        """
        try:
            results = self.model.predict(frame, conf=self.conf_thres, verbose=False)
            detections = []
            
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item()) if box.cls is not None else 0
                        
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class_id': cls_id
                        })
            
            return detections
            
        except Exception as e:
            print(f"检测过程中出现错误: {e}")
            return [] 