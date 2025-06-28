#!/bin/bash
# install_pixhawk_deps.sh
# Jetson Orin Nano PIXHAWK依赖安装脚本

echo "🚁 安装PIXHAWK GPS接收依赖包"
echo "================================"

# 更新包列表
echo "📦 更新包列表..."
sudo apt update

# 安装系统依赖
echo "🔧 安装系统依赖..."
sudo apt install -y python3-pip python3-dev python3-setuptools
sudo apt install -y build-essential cmake
sudo apt install -y libxml2-dev libxslt-dev
sudo apt install -y python3-serial

# 安装Python依赖
echo "🐍 安装Python依赖包..."
pip3 install --user pymavlink
pip3 install --user pyserial
pip3 install --user dronekit

# 设置串口权限
echo "🔐 设置串口权限..."
sudo usermod -a -G dialout $USER
sudo usermod -a -G tty $USER

# 创建udev规则（可选）
echo "📋 创建PIXHAWK设备规则..."
sudo tee /etc/udev/rules.d/99-pixhawk.rules > /dev/null <<EOF
# PIXHAWK设备规则
SUBSYSTEM=="tty", ATTRS{idVendor}=="26ac", ATTRS{idProduct}=="0011", SYMLINK+="pixhawk"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1209", ATTRS{idProduct}=="5740", SYMLINK+="pixhawk"
SUBSYSTEM=="tty", ATTRS{manufacturer}=="ArduPilot", SYMLINK+="pixhawk"
SUBSYSTEM=="tty", ATTRS{manufacturer}=="PX4", SYMLINK+="pixhawk"
EOF

# 重新加载udev规则
sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "✅ 安装完成!"
echo ""
echo "📝 使用说明:"
echo "1. 重新登录或重启系统以应用权限更改"
echo "2. 连接PIXHAWK到Jetson的USB端口"
echo "3. 运行测试脚本: python3 pixhawk_gps_test.py"
echo ""
echo "🔍 常见端口:"
echo "   /dev/ttyUSB0 - USB转串口适配器"
echo "   /dev/ttyACM0 - USB直连设备"
echo "   /dev/pixhawk - 如果设备被正确识别"
echo ""
echo "⚠️  注意事项:"
echo "   - 确保PIXHAWK固件支持MAVLink协议"
echo "   - 检查波特率设置（通常为57600或115200）"
echo "   - 如果连接失败，尝试不同的端口和波特率" 