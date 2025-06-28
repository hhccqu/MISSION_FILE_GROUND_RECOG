#!/usr/bin/env python3
"""
YOLO模型导出为ONNX格式
解决PyTorch和torchvision版本兼容性问题的备用方案
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

def export_to_onnx():
    """导出YOLO模型为ONNX格式"""
    try:
        from ultralytics import YOLO
        print("使用ultralytics导出ONNX模型...")
    except ImportError:
        print("错误: 无法导入ultralytics")
        print("请安装: pip install ultralytics")
        return False
    
    # 查找模型文件
    possible_model_paths = [
        "weights/best.pt",
        "../weights/best.pt",
        "best.pt"
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("错误: 未找到best.pt模型文件")
        print("请确保模型文件位于以下位置之一:")
        for path in possible_model_paths:
            print(f"  - {path}")
        return False
    
    try:
        # 加载模型
        print(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 导出为ONNX
        onnx_path = model_path.replace('.pt', '.onnx')
        print(f"导出ONNX模型到: {onnx_path}")
        
        # 导出参数
        model.export(
            format='onnx',
            imgsz=640,
            optimize=True,
            half=False,  # 在Jetson上建议使用FP32
            simplify=True,
            opset=11,
            verbose=True
        )
        
        print(f"✅ ONNX模型导出成功: {onnx_path}")
        
        # 验证导出的模型
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            print(f"📊 模型大小: {file_size:.2f} MB")
            return True
        else:
            print("❌ 导出失败: 未找到输出文件")
            return False
            
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False

def main():
    print("=" * 50)
    print("YOLO模型ONNX导出工具")
    print("=" * 50)
    
    if export_to_onnx():
        print("\n✅ 导出完成！")
        print("现在您可以使用 inference4_realtime_fixed.py 运行推理")
    else:
        print("\n❌ 导出失败！")
        print("请检查错误信息并重试")

if __name__ == "__main__":
    main() 