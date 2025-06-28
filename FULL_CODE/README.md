# 箭头检测与GPS定位系统

该项目是一个基于计算机视觉和GPS定位的箭头检测系统，专为Jetson Orin Nano平台优化，可以实时检测视频中的箭头标识，识别其上的文字，并结合GPS数据记录位置信息。

## 功能特点

- **目标检测**：使用TensorRT加速的YOLO模型检测箭头标识
- **文字识别**：使用EasyOCR识别箭头上的文字内容
- **方向校正**：基于红色区域的智能箭头方向校正
- **GPS定位**：通过MAVLink协议与Pixhawk飞控通信获取GPS数据
- **数据记录**：将检测结果、GPS信息和图像保存到数据库和文件系统
- **可视化界面**：实时显示检测结果、GPS信息和系统状态

## 系统架构

项目采用模块化设计，主要包含以下组件：

- **GPS接收器**：负责与Pixhawk飞控通信获取GPS数据
- **箭头处理器**：负责箭头图像的方向校正和OCR文字识别
- **数据记录器**：负责检测结果的存储和管理
- **主程序**：整合各模块功能，提供命令行接口

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python main.py
```

这将使用默认参数启动系统，使用摄像头作为视频源。

### 命令行参数

```bash
python main.py --video /path/to/video.mp4 --model-dir /path/to/models --data-dir /path/to/data
```

完整参数列表：

- `--video`：视频源路径，使用数字表示摄像头设备ID（默认：0）
- `--model-dir`：模型目录（默认：/home/lyc/CQU_Ground_ReconnaissanceStrike/weights）
- `--conf-thres`：检测置信度阈值（默认：0.25）
- `--gps-port`：GPS串口设备（默认：/dev/ttyACM0）
- `--gps-baud`：GPS串口波特率（默认：57600）
- `--no-gps`：禁用GPS功能
- `--data-dir`：数据存储目录（默认：/home/lyc/detection_data）
- `--display`：启用图形界面显示（默认：启用）
- `--width`：显示窗口宽度（默认：1280）
- `--height`：显示窗口高度（默认：720）
- `--gpu`：使用GPU进行OCR

### 键盘控制

- `q`：退出程序
- `s`：手动保存当前帧

## 数据存储

检测数据将保存在指定的数据目录中，包含以下内容：

- **images/**：保存裁剪后的箭头图像
- **database/**：SQLite数据库，存储检测记录和GPS信息
- **logs/**：JSON格式的检测记录备份
- **manual_captures/**：手动保存的完整帧图像

## 开发说明

### 目录结构

```
├── main.py                 # 主程序
├── modules/                # 模块目录
│   ├── __init__.py         # 模块包初始化
│   ├── arrow_processor.py  # 箭头处理模块
│   ├── data_recorder.py    # 数据记录模块
│   └── gps_receiver.py     # GPS接收模块
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明
```

### 扩展开发

- **添加新的检测模型**：修改 `main.py` 中的模型加载部分
- **自定义数据记录格式**：修改 `data_recorder.py` 中的数据结构和存储逻辑
- **添加新的通信协议**：在 `gps_receiver.py` 中添加新的通信方式

## 注意事项

- 系统需要安装OpenCV、PyTorch、EasyOCR和PyMAVLink等依赖
- 使用TensorRT加速需要先导出TensorRT引擎文件
- 在Jetson平台上使用EasyOCR时建议使用CPU模式，除非已正确配置CUDA 