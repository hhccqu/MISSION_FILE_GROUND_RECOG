# 无人机打击任务系统使用说明

## 系统概述

本系统实现了完整的无人机对地打击任务流程：
1. **目标数据分析** - 从识别结果中找到数字目标的中位数
2. **MAVLink通信** - 与Pixhawk飞控建立通信连接
3. **任务规划** - 自动生成打击航点并发送给飞控
4. **任务执行** - 实时监控和控制任务执行

## 分析结果

根据当前数据分析：
- **总目标数**: 665个
- **有效数字目标**: 65个  
- **数字分布**:
  - 91: 36次 (55.4%) - 主要目标
  - 95: 16次 (24.6%) - 次要目标
  - 其他: 51, 56, 61, 63, 81, 96
- **中位数目标**: **91**
- **打击目标坐标**: (30.6604264°, 104.1220480°)
- **目标置信度**: 0.731

## 系统组件

### 1. target_median_finder.py
**功能**: 分析目标数据，找到中位数目标
```bash
python target_median_finder.py
```

**输出**:
- 数字分布统计图
- 中位数计算结果
- 最佳打击目标信息

### 2. mavlink_strike_commander.py
**功能**: MAVLink飞控通信模块
```bash
python mavlink_strike_commander.py
```

**功能特性**:
- 连接Pixhawk飞控
- 实时状态监控
- 发送航点任务
- 飞行模式控制
- 解锁/锁定控制

### 3. strike_mission_executor.py
**功能**: 完整任务执行器
```bash
# 交互模式 (默认)
python strike_mission_executor.py

# 自动执行模式
python strike_mission_executor.py auto [高度]

# 仅分析数据
python strike_mission_executor.py analyze

# 测试连接
python strike_mission_executor.py test
```

## 使用方法

### 前置条件

1. **安装依赖**:
```bash
pip install pymavlink statistics
```

2. **硬件连接**:
   - 确保Pixhawk飞控通过USB连接到计算机
   - 检查设备路径 (Windows: COM端口, Linux: /dev/ttyACM0)

3. **数据文件**:
   - 确保 `strike_targets.json` 文件存在
   - 文件包含目标识别和GPS定位数据

### 快速开始

#### 方法1: 自动执行 (推荐)
```bash
python strike_mission_executor.py auto 100
```
- 自动分析目标，连接飞控，发送任务
- 参数100为飞行高度(米)

#### 方法2: 交互模式
```bash
python strike_mission_executor.py
```
然后按提示操作：
1. 输入飞行高度
2. 使用命令控制任务执行

#### 方法3: 分步执行
```bash
# 1. 分析目标数据
python target_median_finder.py

# 2. 测试飞控连接
python mavlink_strike_commander.py

# 3. 执行完整任务
python strike_mission_executor.py
```

### 交互命令

在交互模式下可用的命令：

| 命令 | 功能 | 说明 |
|------|------|------|
| `status` | 显示飞控状态 | 查看GPS、高度、速度等信息 |
| `arm` | 解锁飞控 | 允许电机启动 |
| `disarm` | 锁定飞控 | 停止电机 |
| `auto` | 自动模式 | 切换到自动飞行模式 |
| `manual` | 手动模式 | 切换到手动控制模式 |
| `start` | 启动任务 | 开始执行航点任务 |
| `stop` | 紧急停止 | 立即返航 |
| `quit` | 退出程序 | 断开连接并退出 |

### 任务执行流程

1. **数据分析阶段**:
   ```
   🎯 开始分析目标数据...
   ✅ 成功加载 665 个目标数据
   🎯 找到 65 个有效数字目标
   📈 数字分布统计
   📊 中位数: 91
   ```

2. **飞控连接阶段**:
   ```
   🔗 正在连接飞控: /dev/ttyACM0
   ✅ 飞控连接成功!
   📡 状态监控已启动
   ```

3. **任务准备阶段**:
   ```
   🎯 准备打击任务...
   📍 打击目标: 91 (ID: T0151)
   📡 发送目标航点到飞控...
   ✅ 打击任务准备完成!
   ```

4. **任务执行阶段**:
   ```
   🚀 启动打击任务...
   📊 任务监控
   ✅ 打击任务已启动!
   ```

## 配置参数

### 连接配置
```python
# Windows
mavlink_connection = "COM3"  # 根据设备管理器查看

# Linux
mavlink_connection = "/dev/ttyACM0"  # 或 /dev/ttyUSB0

# 网络连接
mavlink_connection = "udp:127.0.0.1:14550"
```

### 飞行参数
- **默认高度**: 100米
- **最小起飞高度**: 50米
- **盘旋半径**: 50米
- **数据更新频率**: 2Hz

## 安全注意事项

⚠️ **重要安全提醒**:

1. **测试环境**: 首先在仿真环境中测试
2. **飞行空域**: 确保在合法飞行区域内操作
3. **应急准备**: 随时准备手动接管控制
4. **GPS状态**: 确保GPS定位良好 (fix_type >= 3)
5. **电池检查**: 确保电池电量充足
6. **天气条件**: 避免恶劣天气飞行

## 故障排除

### 常见问题

1. **MAVLink连接失败**:
   ```
   ❌ 连接飞控失败: [Errno 2] No such file or directory
   ```
   - 检查设备连接
   - 确认设备路径正确
   - 检查权限设置

2. **GPS定位不良**:
   ```
   ⚠️ GPS定位质量不佳 (fix_type: 1)
   ```
   - 等待GPS信号稳定
   - 移动到开阔区域
   - 检查GPS天线连接

3. **目标数据不足**:
   ```
   ❌ 没有找到有效的数字目标
   ```
   - 检查数据文件完整性
   - 确认OCR识别质量
   - 调整置信度阈值

### 调试模式

启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 自定义目标选择
修改 `target_median_finder.py` 中的选择逻辑：
```python
def find_custom_target(self, criteria):
    # 自定义目标选择算法
    pass
```

### 多目标打击
扩展 `mavlink_strike_commander.py` 支持多航点：
```python
def send_multiple_targets(self, targets):
    # 多目标航点规划
    pass
```

### 实时目标更新
集成实时检测系统：
```python
def update_targets_realtime(self):
    # 实时目标更新
    pass
```

## 技术支持

如有问题，请检查：
1. 系统日志输出
2. 飞控状态信息
3. 网络连接状态
4. 硬件设备状态

---

**版本**: v1.0  
**更新日期**: 2024年12月  
**兼容性**: Python 3.7+, ArduPilot/PX4 