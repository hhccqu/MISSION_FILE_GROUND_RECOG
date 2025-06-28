#!/usr/bin/env python3
# yolo_trt_utils_optimized.py
# 针对Jetson Orin Nano优化的YOLO TensorRT检测器

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
from ultralytics import YOLO
import time
import os
from typing import List, Dict, Tuple, Optional

class JetsonOptimizedYOLODetector:
    """针对Jetson Orin Nano优化的YOLO TensorRT检测器"""
    
    def __init__(self, model_path: str, conf_thres: float = 0.25, 
                 iou_thres: float = 0.45, max_det: int = 300, 
                 device: str = 'cuda:0'):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径 (.pt, .engine, .onnx)
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            max_det: 最大检测数量
            device: 设备
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        
        # TensorRT相关
        self.engine = None
        self.context = None
        self.stream = None
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        
        # 模型信息
        self.input_shape = None
        self.output_shapes = []
        self.class_names = []
        
        # 性能统计
        self.inference_times = []
        self.using_trt = False
        
        # 初始化模型
        self._initialize_model()
        
    def _initialize_model(self):
        """初始化模型"""
        if self.model_path.endswith('.engine'):
            print("🚀 加载TensorRT引擎...")
            self._load_tensorrt_engine()
            self.using_trt = True
        elif self.model_path.endswith('.pt'):
            print("📦 加载PyTorch模型...")
            self._load_pytorch_model()
            # 尝试转换为TensorRT
            self._try_convert_to_tensorrt()
        else:
            raise ValueError(f"不支持的模型格式: {self.model_path}")
    
    def _load_tensorrt_engine(self):
        """加载TensorRT引擎"""
        # 创建TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(self.model_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # 创建CUDA流
        self.stream = cuda.Stream()
        
        # 分配内存
        self._allocate_buffers()
        
        print(f"✅ TensorRT引擎加载成功")
        print(f"   - 输入形状: {self.input_shape}")
        print(f"   - 输出形状: {self.output_shapes}")
    
    def _load_pytorch_model(self):
        """加载PyTorch模型"""
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        
        # 获取类别名称
        if hasattr(self.model.model, 'names'):
            self.class_names = list(self.model.model.names.values())
        
        print(f"✅ PyTorch模型加载成功")
        print(f"   - 类别数量: {len(self.class_names)}")
    
    def _try_convert_to_tensorrt(self):
        """尝试将PyTorch模型转换为TensorRT"""
        try:
            engine_path = self.model_path.replace('.pt', '_orin_fp16.engine')
            
            if not os.path.exists(engine_path):
                print("🔄 转换PyTorch模型为TensorRT引擎...")
                
                # 使用Ultralytics导出
                self.model.export(
                    format='engine',
                    half=True,  # FP16
                    device=self.device,
                    workspace=2,  # 2GB workspace for Orin Nano
                    verbose=True
                )
                
                print(f"✅ TensorRT引擎转换完成: {engine_path}")
            
            # 重新加载TensorRT引擎
            if os.path.exists(engine_path):
                self.model_path = engine_path
                self._load_tensorrt_engine()
                self.using_trt = True
                
        except Exception as e:
            print(f"⚠️  TensorRT转换失败，继续使用PyTorch: {e}")
    
    def _allocate_buffers(self):
        """分配CUDA内存"""
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            binding = self.engine.get_binding_name(i)
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(cuda_mem))
            
            if self.engine.binding_is_input(i):
                self.input_shape = self.engine.get_binding_shape(i)
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.output_shapes.append(self.engine.get_binding_shape(i))
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        if self.using_trt:
            # TensorRT预处理
            input_h, input_w = self.input_shape[2], self.input_shape[3]
        else:
            # PyTorch预处理
            input_h, input_w = 640, 640
        
        # 调整大小并保持宽高比
        img = cv2.resize(image, (input_w, input_h))
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 添加batch维度
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _tensorrt_inference(self, preprocessed_img: np.ndarray) -> List[np.ndarray]:
        """TensorRT推理"""
        # 拷贝输入数据到GPU
        np.copyto(self.host_inputs[0], preprocessed_img.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 拷贝输出数据到CPU
        outputs = []
        for i in range(len(self.host_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.cuda_outputs[i], self.stream)
            outputs.append(self.host_outputs[i].copy())
        
        # 同步
        self.stream.synchronize()
        
        return outputs
    
    def _pytorch_inference(self, image: np.ndarray) -> List[np.ndarray]:
        """PyTorch推理"""
        results = self.model(image, conf=self.conf_thres, iou=self.iou_thres, 
                           max_det=self.max_det, verbose=False)
        return results
    
    def _postprocess_outputs(self, outputs, original_shape) -> List[Dict]:
        """后处理输出"""
        if self.using_trt:
            return self._postprocess_tensorrt_outputs(outputs, original_shape)
        else:
            return self._postprocess_pytorch_outputs(outputs, original_shape)
    
    def _postprocess_tensorrt_outputs(self, outputs, original_shape) -> List[Dict]:
        """TensorRT输出后处理"""
        detections = []
        
        # 重塑输出
        output = outputs[0].reshape(self.output_shapes[0])
        
        # 解析检测结果
        for detection in output[0]:  # batch_size = 1
            if detection[4] > self.conf_thres:  # 置信度过滤
                x1, y1, x2, y2, conf = detection[:5]
                class_id = int(np.argmax(detection[5:]))
                
                # 坐标转换回原图尺寸
                orig_h, orig_w = original_shape[:2]
                input_h, input_w = self.input_shape[2], self.input_shape[3]
                
                x1 = int(x1 * orig_w / input_w)
                y1 = int(y1 * orig_h / input_h)
                x2 = int(x2 * orig_w / input_w)
                y2 = int(y2 * orig_h / input_h)
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if self.class_names else str(class_id)
                })
        
        return detections
    
    def _postprocess_pytorch_outputs(self, results, original_shape) -> List[Dict]:
        """PyTorch输出后处理"""
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id] if self.class_names else str(class_id)
                    })
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        执行目标检测
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表
        """
        start_time = time.time()
        
        # 预处理
        preprocessed_img = self._preprocess_image(image)
        
        # 推理
        if self.using_trt:
            outputs = self._tensorrt_inference(preprocessed_img)
        else:
            outputs = self._pytorch_inference(image)
        
        # 后处理
        detections = self._postprocess_outputs(outputs, image.shape)
        
        # 记录推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        return detections
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if not self.inference_times:
            return {}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'using_tensorrt': self.using_trt,
            'model_path': self.model_path
        }
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.synchronize()


# 兼容性包装器
class YOLOTRTDetector(JetsonOptimizedYOLODetector):
    """兼容原有代码的包装器"""
    def __init__(self, model_path: str, conf_thres: float = 0.25, use_trt: bool = True):
        super().__init__(model_path, conf_thres)
        # 保持兼容性
        pass


# 使用示例
if __name__ == "__main__":
    # 测试检测器
    detector = JetsonOptimizedYOLODetector("../weights/best.pt")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 预热
    print("预热中...")
    for _ in range(5):
        detector.detect(test_image)
    
    # 性能测试
    print("性能测试中...")
    start_time = time.time()
    for _ in range(100):
        detections = detector.detect(test_image)
    
    total_time = time.time() - start_time
    print(f"100次推理总时间: {total_time:.2f}s")
    print(f"平均FPS: {100/total_time:.2f}")
    
    # 显示性能统计
    stats = detector.get_performance_stats()
    print("性能统计:", stats) 