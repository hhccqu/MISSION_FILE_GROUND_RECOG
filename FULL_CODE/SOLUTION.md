# 箭头检测与GPS定位系统 - 模块化重构方案

## 重构概述

原始的`inference4_enhanced.py`文件体量较大，包含多个功能混合在一起，不利于维护和扩展。本次重构将代码按功能模块拆分为多个文件，采用模块化设计，提高代码的可维护性和可扩展性。

## 模块结构

### 1. 主程序模块

- **main.py**: 整合各功能模块，提供命令行接口和主程序流程

### 2. 核心功能模块 (`modules/`)

- **gps_receiver.py**: GPS接收器模块，负责与Pixhawk飞控通信获取GPS数据
  - `GPSData`: GPS数据结构
  - `GPSReceiver`: GPS接收器类

- **arrow_processor.py**: 箭头图像处理模块，负责箭头方向校正和OCR识别
  - `ArrowProcessor`: 箭头处理器类

- **data_recorder.py**: 数据记录模块，负责检测结果的存储和管理
  - `DetectionRecord`: 检测记录数据结构
  - `DataRecorder`: 数据记录器类

- **__init__.py**: 模块包初始化文件，导出主要类和函数

### 3. 工具函数模块 (`utils/`)

- **visualization.py**: 可视化工具函数
  - `draw_detection`: 绘制检测框和文本
  - `draw_gps_info`: 绘制GPS信息
  - `draw_status_bar`: 绘制状态栏

- **conversion.py**: 坐标转换工具函数
  - `pixel_to_geo`: 像素坐标转地理坐标
  - `geo_to_pixel`: 地理坐标转像素坐标

- **__init__.py**: 工具包初始化文件，导出工具函数

### 4. 辅助文件

- **requirements.txt**: 项目依赖列表
- **README.md**: 项目说明文档
- **setup.sh**: 安装脚本，用于在Jetson平台上安装依赖

## 功能增强

相比原始代码，本次重构还增加了以下功能：

1. **命令行参数支持**：使用argparse提供灵活的命令行配置
2. **更完善的数据记录**：添加查询功能和JSON备份
3. **更友好的可视化**：半透明背景和状态栏
4. **坐标转换工具**：提供像素坐标与地理坐标的互相转换
5. **安装脚本**：简化Jetson平台上的部署过程
6. **手动保存功能**：支持按键保存当前帧

## 使用方法

1. 安装依赖：
   ```bash
   bash setup.sh
   ```

2. 运行程序：
   ```bash
   python main.py
   ```

3. 使用命令行参数：
   ```bash
   python main.py --video /path/to/video.mp4 --no-gps --data-dir /path/to/data
   ```

## 未来扩展方向

1. 添加Web界面，实现远程监控
2. 支持多相机输入
3. 添加目标跟踪功能
4. 实现实时地图显示
5. 支持更多类型的飞控通信协议