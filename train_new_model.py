# train_new_model.py
# 快速训练新的YOLO模型

import os
import sys
from ultralytics import YOLO
import torch
import yaml

def main():
    print("🚀 开始训练新的YOLO模型")
    print("="*40)
    
    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 更新数据配置文件
    current_dir = os.getcwd()
    train_path = os.path.join(current_dir, "datasets/train/images").replace('\\', '/')
    valid_path = os.path.join(current_dir, "datasets/valid/images").replace('\\', '/')
    
    data_config = {
        'train': train_path,
        'val': valid_path,
        'nc': 1,
        'names': ['-sign-']
    }
    
    with open("datasets/data_training.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ 数据配置已更新")
    print(f"训练路径: {train_path}")
    print(f"验证路径: {valid_path}")
    
    # 选择批次大小
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 8:
            batch_size = 32
        elif gpu_memory >= 6:
            batch_size = 24
        elif gpu_memory >= 4:
            batch_size = 16
        else:
            batch_size = 8
    else:
        batch_size = 4
    
    print(f"批次大小: {batch_size}")
    
    # 加载预训练模型
    model = YOLO("yolov8n.pt")  # 使用YOLOv8 nano
    print("✅ 模型加载完成")
    
    # 开始训练
    print("\n🎯 开始训练...")
    try:
        results = model.train(
            data="datasets/data_training.yaml",
            epochs=80,                    # 训练轮数
            batch=batch_size,             # 批次大小
            imgsz=640,                    # 图像尺寸
            name='new_detector',          # 项目名称
            project='runs/detect',        # 结果目录
            patience=15,                  # 早停耐心值
            save=True,                    # 保存检查点
            device=device,                # 设备
            workers=4,                    # 数据加载线程
            exist_ok=True,                # 覆盖现有项目
            pretrained=True,              # 使用预训练权重
            optimizer='AdamW',            # 优化器
            verbose=True,                 # 详细输出
            seed=42,                      # 随机种子
            single_cls=True,              # 单类检测
            cos_lr=True,                  # 余弦学习率
            close_mosaic=10,              # 最后10轮关闭马赛克
            amp=True,                     # 混合精度训练
            cache=True,                   # 缓存图像
            lr0=0.01,                     # 初始学习率
            lrf=0.01,                     # 最终学习率
            momentum=0.937,               # 动量
            weight_decay=0.0005,          # 权重衰减
            warmup_epochs=3.0,            # 预热轮数
            box=7.5,                      # 边界框损失权重
            cls=0.5,                      # 分类损失权重
            dfl=1.5,                      # DFL损失权重
            hsv_h=0.015,                  # 色调增强
            hsv_s=0.7,                    # 饱和度增强
            hsv_v=0.4,                    # 明度增强
            translate=0.1,                # 平移增强
            scale=0.5,                    # 缩放增强
            fliplr=0.5,                   # 左右翻转
            mosaic=1.0,                   # 马赛克增强
        )
        
        print("\n🎉 训练完成!")
        print(f"📁 结果保存在: runs/detect/new_detector")
        
        # 复制新模型为best1.pt（不替换原模型）
        import shutil
        best_model = "runs/detect/new_detector/weights/best.pt"
        if os.path.exists(best_model):
            # 确保weights目录存在
            os.makedirs("weights", exist_ok=True)
            
            # 保存新模型为best1.pt
            new_model_path = "weights/best1.pt"
            shutil.copy2(best_model, new_model_path)
            print(f"✅ 新模型已保存为: {new_model_path}")
            print(f"📦 原模型 weights/best.pt 保持不变")
            print("🚀 现在您可以选择使用新模型 best1.pt 进行推理了!")
            
            # 显示模型文件信息
            if os.path.exists("weights/best.pt"):
                old_size = os.path.getsize("weights/best.pt") / 1024 / 1024
                print(f"📊 原模型大小: {old_size:.1f}MB")
            
            new_size = os.path.getsize(new_model_path) / 1024 / 1024
            print(f"📊 新模型大小: {new_size:.1f}MB")
            
        else:
            print(f"❌ 未找到训练好的模型: {best_model}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("请检查数据集路径和配置")

if __name__ == "__main__":
    main() 