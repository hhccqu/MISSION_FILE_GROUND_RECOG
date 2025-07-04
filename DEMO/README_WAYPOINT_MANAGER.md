# MAVLink航点管理器使用说明

## 🎯 功能概述

MAVLink航点管理器是一个完整的无人机航点管理和投水任务执行系统，具备以下核心功能：

### 核心功能
1. **MAVLink通信** - 与飞控建立稳定的双向通信
2. **航点管理** - 下载、修改、上传航点任务
3. **投水点计算** - 根据飞行姿态和目标位置智能计算投水点
4. **实时监控** - 监控飞行状态和任务执行情况
5. **集成检测** - 结合目标检测系统实现自动化任务

## 📋 系统架构

```
mavlink_waypoint_manager.py     # 核心航点管理器
├── MAVLinkWaypointManager     # 主管理类
├── WaypointInfo               # 航点信息数据结构
├── DropPoint                  # 投水点信息数据结构
└── 投水点计算算法              # 姿态补偿和风补偿

integrated_drop_mission.py     # 集成任务系统
├── IntegratedDropMission      # 集成任务管理器
├── 目标检测集成               # 与现有检测系统集成
├── 实时数据处理               # 多线程数据处理
└── 任务执行监控               # 完整的任务生命周期管理
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install pymavlink opencv-python numpy

# 确保YOLO检测模型可用
# 检查 ../weights/best1.pt 或 weights/best1.pt
```

### 2. SITL仿真准备

```bash
# 启动Mission Planner SITL
# 1. 打开Mission Planner
# 2. 选择Simulation选项卡
# 3. 选择飞机类型并启动SITL
# 4. 确保TCP端口14550或5760可用
```

### 3. 基础航点管理

```python
from mavlink_waypoint_manager import MAVLinkWaypointManager

# 创建管理器
manager = MAVLinkWaypointManager("udpin:localhost:14550")

# 连接飞控
if manager.connect():
    # 添加目标点
    success = manager.add_target_and_drop_point(
        target_lat=30.123456,
        target_lon=104.123456,
        target_alt=0.0
    )
    
    if success:
        print("✅ 投水航点添加成功!")
```

### 4. 集成任务系统

```python
from integrated_drop_mission import IntegratedDropMission

# 配置参数
config = {
    'mavlink_connection': 'udpin:localhost:14550',
    'video_source': 'test_video.mp4',
    'auto_drop_enabled': True,
    'min_target_confidence': 0.6
}

# 创建并启动任务
mission = IntegratedDropMission(config)
mission.initialize()
mission.start_mission()
```

## 🛠️ 详细功能说明

### 1. 航点管理功能

#### 下载当前任务
```python
# 请求并下载飞控中的当前任务
success = manager.request_mission_list()
if success:
    print(f"下载了 {len(manager.current_waypoints)} 个航点")
```

#### 插入新航点
```python
# 计算投水点
drop_point = manager.calculate_drop_point(
    target_lat=30.123456,
    target_lon=104.123456,
    target_alt=0.0
)

# 插入航点（在当前航点后）
success = manager.insert_drop_waypoint(drop_point, insert_after_current=True)
```

#### 上传任务
```python
# 上传修改后的任务到飞控
success = manager.upload_mission()
```

### 2. 投水点计算算法

#### 基础计算参数
```python
drop_parameters = {
    'approach_distance': 100.0,      # 接近距离(米)
    'drop_altitude_offset': -10.0,   # 投水高度偏移(米)
    'wind_compensation': True,       # 是否风补偿
    'safety_margin': 20.0,          # 安全边距(米)
}
```

#### 计算过程
1. **方位角计算** - 计算从当前位置到目标的方向
2. **接近距离调整** - 根据高度和速度动态调整
3. **投水点定位** - 在目标前方计算精确投水位置
4. **风补偿** - 根据风速和下降时间计算补偿量
5. **坐标转换** - 精确的地理坐标计算

### 3. 实时数据监控

#### 飞行状态监控
```python
status = manager.get_status()
print(f"位置: {status['position']}")
print(f"姿态: {status['attitude']}")
print(f"当前航点: {status['current_waypoint']}")
```

#### 消息类型处理
- `GLOBAL_POSITION_INT` - GPS位置信息
- `ATTITUDE` - 飞机姿态信息
- `WIND` - 风速信息
- `MISSION_CURRENT` - 当前航点状态

### 4. 集成检测功能

#### 自动目标处理
```python
# 检测到目标后自动处理
def _process_target_for_drop(self, target_info):
    gps_pos = target_info['gps_pos']
    
    # 自动添加投水航点
    success = self.waypoint_manager.add_target_and_drop_point(
        gps_pos['latitude'],
        gps_pos['longitude'],
        0.0
    )
```

#### 智能过滤条件
- **置信度阈值** - 只处理高置信度目标
- **冷却时间** - 避免频繁投水
- **最大目标数** - 限制单次任务目标数量

## 🎮 交互操作

### 1. 命令行交互（航点管理器）

```bash
python mavlink_waypoint_manager.py
```

**可用命令：**
- `add <lat> <lon> [alt]` - 添加目标点
- `status` - 显示飞行状态
- `mission` - 显示当前任务
- `quit` - 退出程序

### 2. 实时控制（集成系统）

```bash
python integrated_drop_mission.py
```

**控制按键：**
- `'q'` - 退出任务
- `'s'` - 保存数据
- `'p'` - 暂停/恢复自动投水
- `'t'` - 显示目标统计
- `'m'` - 显示任务状态

## ⚙️ 配置参数

### 1. 连接配置

```python
# UDP连接（推荐）
'mavlink_connection': 'udpin:localhost:14550'

# TCP连接
'mavlink_connection': 'tcp:localhost:5760'

# 串口连接
'mavlink_connection': '/dev/ttyUSB0:57600'
```

### 2. 投水参数

```python
drop_parameters = {
    'approach_distance': 100.0,      # 基础接近距离
    'drop_altitude_offset': -10.0,   # 高度偏移
    'wind_compensation': True,       # 风补偿开关
    'safety_margin': 20.0,          # 安全边距
}
```

### 3. 检测参数

```python
config = {
    'conf_threshold': 0.25,          # YOLO检测阈值
    'min_target_confidence': 0.6,   # 投水目标最小置信度
    'drop_cooldown': 30.0,          # 投水冷却时间
    'max_targets_per_mission': 10,   # 最大目标数
}
```

## 📊 数据格式

### 1. 航点数据结构

```python
@dataclass
class WaypointInfo:
    seq: int                    # 序号
    frame: int                  # 坐标系
    command: int                # 命令类型
    current: int                # 是否为当前航点
    autocontinue: int           # 自动继续
    param1: float               # 参数1
    param2: float               # 参数2
    param3: float               # 参数3
    param4: float               # 参数4（偏航角）
    x: float                    # 纬度
    y: float                    # 经度
    z: float                    # 高度
```

### 2. 投水点数据结构

```python
@dataclass
class DropPoint:
    target_lat: float           # 目标纬度
    target_lon: float           # 目标经度
    drop_lat: float             # 投水纬度
    drop_lon: float             # 投水经度
    drop_altitude: float        # 投水高度
    approach_distance: float    # 接近距离
    wind_compensation: Tuple[float, float]  # 风补偿
```

### 3. 保存的目标数据

```json
{
  "target_id": "T0001",
  "pixel_position": [640, 360],
  "gps_position": {
    "latitude": 30.123456,
    "longitude": 104.123456
  },
  "confidence": 0.85,
  "timestamp": 1703123456.789,
  "flight_data": {
    "latitude": 30.123400,
    "longitude": 104.123400,
    "altitude": 500.0,
    "pitch": -10.5,
    "roll": 1.2,
    "yaw": 90.3,
    "ground_speed": 30.0,
    "heading": 90.0
  }
}
```

## 🔧 高级功能

### 1. 自定义投水命令

```python
# 修改投水命令类型
drop_cmd_wp = WaypointInfo(
    command=mavutil.mavlink.MAV_CMD_DO_SET_SERVO,  # 舵机命令
    param1=9,     # 舵机通道
    param2=1500,  # PWM值
    # ... 其他参数
)
```

### 2. 多目标批量处理

```python
targets = [
    (30.123456, 104.123456),
    (30.124456, 104.124456),
    (30.125456, 104.125456)
]

for lat, lon in targets:
    manager.add_target_and_drop_point(lat, lon)
```

### 3. 实时参数调整

```python
# 动态调整投水参数
manager.drop_parameters['approach_distance'] = 150.0
manager.drop_parameters['wind_compensation'] = False
```

## 🐛 故障排除

### 1. 连接问题

```bash
# 测试连接
python test_sitl_connection.py

# 强制连接测试
python force_tcp_test.py

# 全面连接测试
python final_sitl_solution.py
```

### 2. 常见错误

#### MAVLink连接失败
```
❌ 连接失败: [Errno 111] Connection refused
```
**解决方案：**
- 检查SITL是否正在运行
- 确认端口号正确
- 尝试不同的连接字符串

#### 任务上传失败
```
❌ 任务上传失败或未确认
```
**解决方案：**
- 检查飞控状态
- 确认任务格式正确
- 重试上传操作

#### GPS数据不可用
```
❌ 无法获取当前飞行数据
```
**解决方案：**
- 等待GPS定位稳定
- 检查数据流请求
- 确认消息接收正常

### 3. 性能优化

- 调整数据更新频率
- 优化检测阈值
- 减少不必要的计算
- 使用多线程处理

## 📈 扩展开发

### 1. 自定义投水逻辑

```python
class CustomDropLogic:
    def calculate_custom_drop_point(self, target, flight_data):
        # 自定义投水点计算逻辑
        pass
```

### 2. 集成其他传感器

```python
def integrate_lidar_data(self, lidar_data):
    # 集成激光雷达数据
    pass

def integrate_weather_data(self, weather_data):
    # 集成气象数据
    pass
```

### 3. 任务规划优化

```python
def optimize_mission_path(self, targets):
    # 优化多目标访问路径
    pass
```

## 🎯 使用场景

1. **森林消防** - 精确投水灭火
2. **农业植保** - 定点药物投放
3. **应急救援** - 物资精确投递
4. **环境监测** - 采样设备投放
5. **科学研究** - 实验设备部署

## 📞 技术支持

如有问题，请检查：
1. 系统日志输出
2. MAVLink消息流
3. 飞控状态指示
4. 网络连接状态
5. 配置参数设置

---

**注意：** 本系统涉及无人机飞行控制，使用前请确保：
- 在合法飞行区域内操作
- 具备相应的飞行资质
- 做好安全防护措施
- 进行充分的地面测试 