# train_yolo_improved.py
# 改进的YOLO训练脚本 - 针对地面目标识别优化

import os
import sys
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def check_environment():
    """检查训练环境"""
    print("🔍 检查训练环境...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
    
    # 检查数据集
    data_yaml = "datasets/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ 数据配置文件不存在: {data_yaml}")
        return False
    
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 检查训练和验证目录
    train_images = Path("datasets/train/images")
    valid_images = Path("datasets/valid/images")
    
    if train_images.exists():
        train_count = len(list(train_images.glob("*")))
        print(f"✅ 训练图像: {train_count}张")
    else:
        print("❌ 训练图像目录不存在")
        return False
    
    if valid_images.exists():
        valid_count = len(list(valid_images.glob("*")))
        print(f"✅ 验证图像: {valid_count}张")
    else:
        print("❌ 验证图像目录不存在")
        return False
    
    print(f"✅ 数据集检查完成")
    return True

def update_data_yaml():
    """更新数据配置文件路径"""
    print("🔧 更新数据配置文件...")
    
    # 获取绝对路径
    current_dir = os.getcwd()
    train_path = os.path.join(current_dir, "datasets/train/images")
    valid_path = os.path.join(current_dir, "datasets/valid/images")
    
    # 创建新的配置
    data_config = {
        'train': train_path.replace('\\', '/'),
        'val': valid_path.replace('\\', '/'),
        'nc': 1,
        'names': ['-sign-']
    }
    
    # 保存配置文件
    with open("datasets/data_updated.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ 数据配置已更新: datasets/data_updated.yaml")
    return "datasets/data_updated.yaml"

def get_optimal_batch_size():
    """根据显存自动选择批次大小"""
    if not torch.cuda.is_available():
        return 4  # CPU模式使用小批次
    
    # 获取显存大小（GB）
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_memory >= 8:
        return 32
    elif gpu_memory >= 6:
        return 24
    elif gpu_memory >= 4:
        return 16
    else:
        return 8

def train_improved_model():
    """训练改进的YOLO模型"""
    print("🚀 开始训练改进的YOLO模型")
    print("="*50)
    
    # 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请检查数据集配置")
        return
    
    # 更新数据配置
    data_yaml = update_data_yaml()
    
    # 自动选择批次大小
    batch_size = get_optimal_batch_size()
    print(f"🎯 自动选择批次大小: {batch_size}")
    
    # 选择基础模型
    base_models = ["yolov8n.pt", "yolov8s.pt"]
    selected_model = "yolov8n.pt"  # 默认使用nano版本
    
    for model_path in base_models:
        if os.path.exists(model_path):
            selected_model = model_path
            break
    
    print(f"📦 使用基础模型: {selected_model}")
    
    # 加载模型
    model = YOLO(selected_model)
    
    # 训练参数配置
    train_args = {
        'data': data_yaml,
        'epochs': 100,                    # 增加训练轮数
        'batch': batch_size,              # 自动批次大小
        'imgsz': 640,                     # 图像尺寸
        'name': 'improved_detector',      # 项目名称
        'project': 'runs/detect',         # 结果目录
        'patience': 20,                   # 早停耐心值
        'save': True,                     # 保存检查点
        'save_period': 10,                # 每10轮保存一次
        'cache': True,                    # 缓存图像加速训练
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 8,                     # 数据加载线程数
        'exist_ok': True,                 # 覆盖现有项目
        'pretrained': True,               # 使用预训练权重
        'optimizer': 'AdamW',             # 优化器
        'verbose': True,                  # 详细输出
        'seed': 42,                       # 随机种子
        'deterministic': True,            # 确定性训练
        'single_cls': True,               # 单类检测
        'rect': False,                    # 矩形训练
        'cos_lr': True,                   # 余弦学习率调度
        'close_mosaic': 10,               # 最后10轮关闭马赛克增强
        'resume': False,                  # 不恢复训练
        'amp': True,                      # 混合精度训练
        'fraction': 1.0,                  # 使用全部数据
        'profile': False,                 # 不进行性能分析
        'lr0': 0.01,                      # 初始学习率
        'lrf': 0.01,                      # 最终学习率比例
        'momentum': 0.937,                # 动量
        'weight_decay': 0.0005,           # 权重衰减
        'warmup_epochs': 3.0,             # 预热轮数
        'warmup_momentum': 0.8,           # 预热动量
        'warmup_bias_lr': 0.1,            # 预热偏置学习率
        'box': 7.5,                       # 边界框损失权重
        'cls': 0.5,                       # 分类损失权重
        'dfl': 1.5,                       # DFL损失权重
        'pose': 12.0,                     # 姿态损失权重
        'kobj': 2.0,                      # 关键点对象损失权重
        'label_smoothing': 0.0,           # 标签平滑
        'nbs': 64,                        # 标准批次大小
        'hsv_h': 0.015,                   # 色调增强
        'hsv_s': 0.7,                     # 饱和度增强
        'hsv_v': 0.4,                     # 明度增强
        'degrees': 0.0,                   # 旋转角度
        'translate': 0.1,                 # 平移比例
        'scale': 0.5,                     # 缩放比例
        'shear': 0.0,                     # 剪切角度
        'perspective': 0.0,               # 透视变换
        'flipud': 0.0,                    # 上下翻转概率
        'fliplr': 0.5,                    # 左右翻转概率
        'mosaic': 1.0,                    # 马赛克增强概率
        'mixup': 0.0,                     # 混合增强概率
        'copy_paste': 0.0,                # 复制粘贴增强概率
    }
    
    print("🎯 训练参数配置:")
    for key, value in train_args.items():
        print(f"   {key}: {value}")
    
    print("\n🚀 开始训练...")
    try:
        # 开始训练
        results = model.train(**train_args)
        
        print("\n✅ 训练完成!")
        print(f"📁 结果保存在: runs/detect/improved_detector")
        print(f"🏆 最佳模型: runs/detect/improved_detector/weights/best.pt")
        
        # 显示训练结果
        if hasattr(results, 'results_dict'):
            print("\n📊 训练结果:")
            for metric, value in results.results_dict.items():
                print(f"   {metric}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False

def copy_best_model():
    """复制最佳模型到weights目录"""
    best_model_path = "runs/detect/improved_detector/weights/best.pt"
    target_path = "weights/best1.pt"  # 改为best1.pt
    
    if os.path.exists(best_model_path):
        import shutil
        os.makedirs("weights", exist_ok=True)
        shutil.copy2(best_model_path, target_path)
        print(f"✅ 新模型已保存为: {target_path}")
        
        # 显示原模型信息（不备份不替换）
        old_model = "weights/best.pt"
        if os.path.exists(old_model):
            old_size = os.path.getsize(old_model) / 1024 / 1024
            print(f"📦 原模型 {old_model} 保持不变 (大小: {old_size:.1f}MB)")
        
        # 显示新模型信息
        new_size = os.path.getsize(target_path) / 1024 / 1024
        print(f"📊 新模型大小: {new_size:.1f}MB")
        print("🚀 现在您可以选择使用新模型 best1.pt 进行推理了!")
        
        return True
    else:
        print(f"❌ 未找到训练好的模型: {best_model_path}")
        return False

def main():
    """主函数"""
    print("🎯 YOLO模型重新训练工具")
    print("="*50)
    
    # 训练模型
    if train_improved_model():
        print("\n🎉 训练成功完成!")
        
        # 复制模型
        if copy_best_model():
            print("\n✅ 模型部署完成!")
            print("现在可以使用新训练的模型进行推理了")
        else:
            print("\n⚠️  模型复制失败，请手动复制模型文件")
    else:
        print("\n❌ 训练失败，请检查错误信息")

if __name__ == "__main__":
    main() 