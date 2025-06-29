# 无人机地面目标识别与打击任务系统

## 项目简介

这是一个基于YOLO目标检测和MAVLink协议的无人机自主打击任务系统。系统集成了计算机视觉、无人机控制、目标识别和任务执行等功能，能够实现对地面目标的自动识别、跟踪和精确打击。

## 主要功能

### 🎯 目标检测与识别
- 基于YOLOv8的实时目标检测
- 支持多类别目标识别
- 图像旋转校正和OCR文字识别
- 自定义数据集训练和模型优化

### 🚁 无人机控制
- MAVLink协议通信
- SITL仿真环境支持
- 自主飞行路径规划
- 实时位置和状态监控

### ⚡ 打击任务执行
- 自动目标锁定
- 精确坐标计算
- 任务状态实时反馈
- 多目标优先级管理

### 🔧 系统集成
- TCP/UDP通信测试
- 模块化架构设计
- 配置文件管理
- 日志记录和调试

## 项目结构

```
MISSION_FILE_GROUND_RECOG/
├── DEMO/                          # 演示和测试脚本
│   ├── force_tcp_test.py         # TCP连接强制测试
│   ├── simple_sitl_test.py       # SITL简单测试
│   ├── test_sitl_connection.py   # SITL连接测试
│   ├── final_sitl_solution.py    # 最终SITL解决方案
│   ├── sitl_strike_mission.py    # SITL打击任务
│   ├── strike_mission_system.py  # 打击任务系统
│   └── run_strike_mission.py     # 运行打击任务
├── DEMO_DETECT_TEST/              # 检测测试相关
├── FULL_CODE/                     # 完整代码库
├── CQUFLY/                        # 重庆大学飞行相关
├── config/                        # 配置文件
│   ├── camera_intrinsics.yaml    # 相机标定参数
│   └── flight_params.yaml        # 飞行参数配置
├── datasets/                      # 数据集
│   ├── train/                     # 训练数据
│   ├── valid/                     # 验证数据
│   └── data.yaml                  # YOLO训练配置
├── scripts/                       # 脚本文件
├── weights/                       # 模型权重
├── ready/                         # 处理后的图像
├── TensorRT/                      # TensorRT优化
├── runs/                          # 训练运行记录
├── images/                        # 图像文件
├── sitl_strike_mission.py         # 主要打击任务脚本
├── train_new_model.py             # 模型训练脚本
├── strike_targets.json            # 目标数据
├── requirements.txt               # 依赖包列表
└── README.md                      # 项目说明
```

## 安装与配置

### 环境要求
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- PyMAVLink
- NumPy, Pandas等

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/YOUR_USERNAME/MISSION_FILE_GROUND_RECOG.git
cd MISSION_FILE_GROUND_RECOG
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置参数
编辑 `config/` 目录下的配置文件

### SITL仿真环境配置
1. 安装Mission Planner或QGroundControl
2. 配置SITL参数
3. 运行连接测试：
```bash
python DEMO/force_tcp_test.py
```

## 使用方法

### 1. 目标检测训练
```bash
python train_new_model.py
```

### 2. SITL连接测试
```bash
python DEMO/simple_sitl_test.py
```

### 3. 运行打击任务
```bash
python sitl_strike_mission.py
```

### 4. 完整任务系统
```bash
python DEMO/run_strike_mission.py
```

## 核心模块说明

### TCP连接测试 (`force_tcp_test.py`)
- 原始TCP数据接收测试
- MAVLink强制连接测试
- UDP监听模式测试
- 多种连接方式尝试

### 打击任务系统 (`strike_mission_system.py`)
- 目标检测和识别
- 坐标转换和定位
- 飞行路径规划
- 任务状态管理

### SITL集成 (`final_sitl_solution.py`)
- SITL环境初始化
- MAVLink通信建立
- 飞行控制命令
- 实时状态监控

## 技术特性

- **实时性能**：优化的检测算法，支持实时视频流处理
- **高精度**：精确的目标定位和坐标转换
- **可扩展性**：模块化设计，易于功能扩展
- **稳定性**：完善的错误处理和异常恢复机制
- **兼容性**：支持多种无人机平台和地面站软件

## 开发团队

重庆大学航空建模团队

## 许可证

本项目采用MIT许可证

## 更新日志

- **v1.0** - 基础目标检测和SITL集成
- **v1.1** - 添加TCP连接强制测试功能
- **v1.2** - 完善打击任务系统
- **v1.3** - 优化SITL解决方案

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 联系方式

如有问题请提交Issue或联系开发团队。

