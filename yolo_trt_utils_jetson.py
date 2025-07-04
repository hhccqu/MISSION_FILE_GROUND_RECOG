#!/usr/bin/env python3
# yolo_trt_utils_jetson.py
# Jetson优化的YOLO TensorRT检测器

import os
import time
import numpy as np
from ultralytics import YOLO
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class YOLOTRTDetectorJetson:
    """Jetson优化的YOLO TensorRT检测器"""
    
    def __init__(self, model_path="weights/best1.pt", conf_thres=0.25, use_trt=True):
        """
        初始化Jetson优化的YOLO检测器
        
        参数:
            model_path: 模型文件路径
            conf_thres: 置信度阈值
            use_trt: 是否使用TensorRT加速
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.use_trt = use_trt
        
        # 检查是否有TensorRT引擎文件
        engine_path = model_path.replace('.pt', '.engine')
        if use_trt and os.path.exists(engine_path):
            self.model_path = engine_path
            self.using_trt = True
            print(f"✅ 使用TensorRT引擎: {engine_path}")
        else:
            self.using_trt = False
            print(f"📝 使用PyTorch模型: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载模型
        try:
            self.model = YOLO(self.model_path)
            
            # Jetson特定优化
            if self.using_trt:
                # TensorRT引擎已经优化
                print("🚀 TensorRT引擎加载完成")
            else:
                # PyTorch模型优化
                self.model.to('cuda')  # 确保在GPU上运行
                print("🎯 PyTorch模型加载完成，运行在GPU上")
                
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
        
        # 性能统计
        self.inference_times = []
        self.warmup_done = False
    
    def warmup(self, warmup_frames=5):
        """模型预热，提高首次推理性能"""
        if self.warmup_done:
            return
            
        print("🔥 模型预热中...")
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        for i in range(warmup_frames):
            _ = self.model.predict(dummy_frame, conf=self.conf_thres, verbose=False)
            print(f"预热进度: {i+1}/{warmup_frames}")
        
        self.warmup_done = True
        print("✅ 模型预热完成")
    
    def detect(self, frame):
        """
        对单帧图像进行检测（Jetson优化版本）
        
        参数:
            frame: 输入图像
            
        返回:
            检测结果列表
        """
        if not self.warmup_done:
            self.warmup()
        
        start_time = time.time()
        
        try:
            # Jetson优化的推理配置
            results = self.model.predict(
                frame, 
                conf=self.conf_thres, 
                verbose=False,
                device='cuda',  # 强制使用GPU
                half=True,      # 使用FP16精度提高性能
                max_det=100     # 限制最大检测数量
            )
            
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
            
            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # 保持最近100次推理时间记录
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            return detections
            
        except Exception as e:
            print(f"检测过程中出现错误: {e}")
            return []
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.inference_times:
            return {}
        
        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'using_tensorrt': self.using_trt,
            'total_inferences': len(self.inference_times)
        }
    
    def optimize_for_jetson(self):
        """Jetson特定优化设置"""
        # 设置CUDA优化
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        
        # 设置OpenCV优化
        import cv2
        cv2.setUseOptimized(True)
        cv2.setNumThreads(6)  # 使用所有CPU核心
        
        print("⚡ Jetson优化设置完成") 