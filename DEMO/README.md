# 无人机对地打击任务系统

## 🎯 系统概述

这是一个完整的无人机对地打击任务系统，集成了目标检测、OCR识别、GPS定位和数据分析功能。系统能够实时识别地面箭头目标，提取其中的数字信息，并计算目标的实际GPS坐标。

## 🏗️ 系统架构

```
DEMO/
├── inference4_realtime.py          # 原始实时检测脚本
├── yolo_trt_utils.py              # YOLO检测器工具
├── target_geo_calculator.py       # 地理坐标计算模块
├── strike_mission_system.py       # 主任务系统
├── run_strike_mission.py          # 快速启动脚本
├── target_data_analyzer.py        # 数据分析工具
└── README.md                      # 使用说明
```

## 🚀 核心功能

### 1. 目标检测与识别
- **YOLO多目标检测**：同时检测多个箭头目标
- **图像旋转校正**：基于红色区域的自动旋转校正
- **OCR数字识别**：提取箭头中的二位数字
- **置信度筛选**：过滤低置信度检测结果

### 2. 飞行数据模拟
- **GPS轨迹模拟**：模拟直线飞行轨迹
- **姿态角度模拟**：俯仰角、横滚角、偏航角
- **实时飞行状态**：高度、速度、航向等

### 3. 地理坐标计算
- **像素到GPS转换**：根据相机参数和飞行姿态计算目标GPS坐标
- **多参数校正**：考虑相机视场角、飞机姿态、高度等因素
- **坐标系转换**：图像坐标系到地理坐标系的精确转换

### 4. 数据管理
- **实时数据保存**：JSON格式保存所有目标信息
- **结构化存储**：包含目标ID、识别数字、GPS坐标、飞行数据等
- **自动备份**：每100个目标自动保存

## 📋 安装要求

### 必需依赖
```bash
pip install opencv-python
pip install ultralytics
pip install easyocr
pip install numpy
```

### 可选依赖（用于数据分析）
```bash
pip install matplotlib
pip install pandas
```

### 模型文件
确保以下路径之一存在YOLO模型：
- `../weights/best1.pt`
- `weights/best1.pt`
- `../weights/best_trt.engine`
- `weights/best_trt.engine`

## 🎮 使用方法

### 方法1：快速启动
```bash
cd DEMO
python run_strike_mission.py
```

### 方法2：自定义配置
```python
from strike_mission_system import StrikeMissionSystem

# 自定义配置
config = {
    'conf_threshold': 0.25,
    'start_lat': 30.6586,      # 起始纬度
    'start_lon': 104.0647,     # 起始经度
    'altitude': 500.0,         # 飞行高度（米）
    'speed': 30.0,             # 飞行速度（m/s）
    'heading': 90.0,           # 航向角（度）
    'save_file': 'my_targets.json'
}

# 创建并运行任务
mission = StrikeMissionSystem(config)
mission.initialize()
mission.run_video_mission("video.mp4")
```

## 🎛️ 操作控制

### 实时控制按键
- **'q'** - 退出任务
- **'s'** - 立即保存当前数据
- **'r'** - 重置统计信息
- **'c'** - 清空目标数据

### 显示信息
系统实时显示以下信息：
- 飞行GPS坐标和高度
- 飞机姿态角度（俯仰、横滚、偏航）
- 飞行速度和航向
- 检测统计（帧数、FPS、目标数量）
- 每个目标的ID、识别数字、GPS坐标、置信度

## 📊 数据分析

### 运行分析工具
```bash
python target_data_analyzer.py
```

### 分析功能
1. **基础统计**：目标总数、识别成功率、置信度分布
2. **空间分布**：GPS坐标范围、覆盖区域面积
3. **飞行数据**：高度、速度、姿态角度统计
4. **距离分析**：目标间距离分布
5. **可视化地图**：目标分布图（需matplotlib）
6. **KML导出**：Google Earth格式文件

### 输出文件
- `target_analysis_report.txt` - 详细分析报告
- `target_map.png` - 目标分布图
- `strike_targets.kml` - Google Earth文件

## 📁 数据格式

### 目标数据结构（JSON）
```json
{
  "target_id": "T0001",
  "detected_number": "25",
  "pixel_position": {"x": 640, "y": 360},
  "confidence": 0.85,
  "gps_position": {
    "latitude": 30.658600,
    "longitude": 104.064700
  },
  "flight_data": {
    "timestamp": 1703123456.789,
    "latitude": 30.658500,
    "longitude": 104.064600,
    "altitude": 500.0,
    "pitch": -10.5,
    "roll": 1.2,
    "yaw": 90.3,
    "ground_speed": 30.0,
    "heading": 90.0
  },
  "detection_timestamp": 1703123456.789
}
```

## ⚙️ 配置参数

### 检测参数
- `conf_threshold`: YOLO检测置信度阈值（默认0.25）
- `min_confidence`: 目标最小置信度（默认0.5）
- `max_targets_per_frame`: 每帧最大处理目标数（默认5）

### 相机参数
- `camera_fov_h`: 水平视场角（默认60°）
- `camera_fov_v`: 垂直视场角（默认45°）

### 飞行参数
- `start_lat/start_lon`: 起始GPS坐标
- `altitude`: 飞行高度（米）
- `speed`: 飞行速度（m/s）
- `heading`: 航向角（度，0=北，90=东）

### 处理参数
- `ocr_interval`: OCR处理间隔（帧数，默认5）
- `save_file`: 数据保存文件名

## 🔧 系统扩展

### 1. 真实飞控接入
替换 `GPSSimulator` 为真实的MAVLink飞控数据读取：
```python
# 示例：接入真实飞控
from pymavlink import mavutil

class RealFlightData:
    def __init__(self, connection_string):
        self.master = mavutil.mavlink_connection(connection_string)
    
    def get_current_position(self):
        # 读取真实飞控数据
        pass
```

### 2. 相机标定
使用真实相机参数替换默认视场角：
```python
# 相机标定参数
config['camera_fov_h'] = 实际水平视场角
config['camera_fov_v'] = 实际垂直视场角
```

### 3. 目标类型扩展
修改YOLO模型支持更多目标类型，扩展OCR识别规则。

## 🐛 常见问题

### 1. 模型文件未找到
确保YOLO模型文件在正确路径，或修改 `_find_model()` 方法中的路径。

### 2. OCR识别失败
- 检查EasyOCR安装是否正确
- 调整图像预处理参数
- 确保目标图像质量足够

### 3. GPS坐标计算异常
- 检查相机视场角参数是否正确
- 确保飞行高度合理（建议100-1000米）
- 验证飞机姿态角度范围

### 4. 性能问题
- 降低 `max_targets_per_frame` 参数
- 增加 `ocr_interval` 间隔
- 使用TensorRT模型加速

