#!/usr/bin/env python3
# quick_env_check.py
# Jetson环境快速检查脚本

import sys
import os
import subprocess
import importlib

def colored_print(text, color='white'):
    """彩色打印"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['end']}")

def check_module(module_name, required=True):
    """检查Python模块是否可用"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        else:
            version = "未知版本"
        colored_print(f"  ✓ {module_name}: {version}", 'green')
        return True, module
    except ImportError:
        status = "❌" if required else "⚠️"
        color = "red" if required else "yellow"
        colored_print(f"  {status} {module_name}: 未安装", color)
        return False, None

def check_system_info():
    """检查系统信息"""
    colored_print("=== 系统信息 ===", 'blue')
    
    # Python版本
    colored_print(f"Python版本: {sys.version.split()[0]}", 'cyan')
    
    # 检查是否在Jetson上
    if os.path.exists('/etc/nv_tegra_release'):
        with open('/etc/nv_tegra_release', 'r') as f:
            jetpack_info = f.read().strip()
        colored_print(f"JetPack信息: {jetpack_info}", 'cyan')
        
        # 获取设备型号
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip().replace('\x00', '')
            colored_print(f"设备型号: {model}", 'cyan')
        except:
            colored_print("设备型号: 无法获取", 'yellow')
    else:
        colored_print("⚠️ 不在Jetson设备上运行", 'yellow')
    
    # 内存信息
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            total_mem = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) // 1024
            available_mem = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) // 1024
        colored_print(f"内存: {available_mem}MB / {total_mem}MB 可用", 'cyan')
    except:
        colored_print("内存信息: 无法获取", 'yellow')
    
    # Swap信息
    try:
        result = subprocess.run(['free', '-m'], capture_output=True, text=True)
        swap_line = [line for line in result.stdout.split('\n') if 'Swap:' in line][0]
        swap_total = swap_line.split()[1]
        swap_used = swap_line.split()[2]
        colored_print(f"Swap: {swap_used}MB / {swap_total}MB 已使用", 'cyan')
    except:
        colored_print("Swap信息: 无法获取", 'yellow')

def check_performance_mode():
    """检查性能模式"""
    colored_print("\n=== 性能模式 ===", 'blue')
    
    try:
        # 检查nvpmodel
        result = subprocess.run(['sudo', 'nvpmodel', '-q'], capture_output=True, text=True)
        if result.returncode == 0:
            colored_print(f"功耗模式: {result.stdout.strip()}", 'cyan')
        else:
            colored_print("无法获取功耗模式信息", 'yellow')
    except:
        colored_print("nvpmodel命令不可用", 'yellow')
    
    # 检查CPU调度器
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
            governor = f.read().strip()
        colored_print(f"CPU调度器: {governor}", 'cyan')
    except:
        colored_print("无法获取CPU调度器信息", 'yellow')

def check_cuda():
    """检查CUDA环境"""
    colored_print("\n=== CUDA环境 ===", 'blue')
    
    # 检查CUDA路径
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if os.path.exists(cuda_home):
        colored_print(f"  ✓ CUDA路径: {cuda_home}", 'green')
    else:
        colored_print(f"  ❌ CUDA路径不存在: {cuda_home}", 'red')
    
    # 检查nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
            colored_print(f"  ✓ NVCC: {version_line.strip()}", 'green')
        else:
            colored_print("  ❌ nvcc命令不可用", 'red')
    except:
        colored_print("  ❌ nvcc命令未找到", 'red')

def check_python_packages():
    """检查Python包"""
    colored_print("\n=== Python包检查 ===", 'blue')
    
    # 核心包
    colored_print("核心依赖:", 'purple')
    numpy_ok, numpy = check_module('numpy')
    scipy_ok, scipy = check_module('scipy', False)
    cv2_ok, cv2 = check_module('cv2')
    
    # 深度学习框架
    colored_print("\n深度学习框架:", 'purple')
    torch_ok, torch = check_module('torch')
    torchvision_ok, torchvision = check_module('torchvision')
    tensorrt_ok, tensorrt = check_module('tensorrt', False)
    
    # 专用库
    colored_print("\n专用库:", 'purple')
    easyocr_ok, easyocr = check_module('easyocr')
    ultralytics_ok, ultralytics = check_module('ultralytics', False)
    pycuda_ok, pycuda = check_module('pycuda', False)
    
    # 其他依赖
    colored_print("\n其他依赖:", 'purple')
    pil_ok, pil = check_module('PIL')
    pymavlink_ok, pymavlink = check_module('pymavlink', False)
    
    return {
        'torch': torch,
        'cv2': cv2,
        'easyocr': easyocr,
        'tensorrt': tensorrt
    }

def check_torch_cuda(torch_module):
    """检查PyTorch CUDA支持"""
    if torch_module is None:
        return
        
    colored_print("\n=== PyTorch CUDA ===", 'blue')
    
    try:
        cuda_available = torch_module.cuda.is_available()
        if cuda_available:
            colored_print("  ✓ CUDA可用", 'green')
            device_count = torch_module.cuda.device_count()
            colored_print(f"  ✓ GPU设备数量: {device_count}", 'green')
            
            if device_count > 0:
                device_name = torch_module.cuda.get_device_name(0)
                colored_print(f"  ✓ GPU设备: {device_name}", 'green')
                
                # GPU内存
                props = torch_module.cuda.get_device_properties(0)
                total_memory = props.total_memory / (1024**3)
                colored_print(f"  ✓ GPU内存: {total_memory:.1f}GB", 'green')
                
                # 简单推理测试
                try:
                    x = torch_module.randn(1, 3, 224, 224).cuda()
                    y = torch_module.nn.functional.relu(x)
                    colored_print("  ✓ GPU推理测试通过", 'green')
                except Exception as e:
                    colored_print(f"  ❌ GPU推理测试失败: {e}", 'red')
        else:
            colored_print("  ❌ CUDA不可用", 'red')
    except Exception as e:
        colored_print(f"  ❌ CUDA检查失败: {e}", 'red')

def check_opencv_cuda(cv2_module):
    """检查OpenCV CUDA支持"""
    if cv2_module is None:
        return
        
    colored_print("\n=== OpenCV CUDA ===", 'blue')
    
    try:
        cuda_devices = cv2_module.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            colored_print(f"  ✓ OpenCV CUDA设备: {cuda_devices}", 'green')
        else:
            colored_print("  ⚠️ OpenCV CUDA设备: 0 (可能未启用CUDA支持)", 'yellow')
    except:
        colored_print("  ❌ OpenCV CUDA不可用", 'red')

def check_camera():
    """检查摄像头"""
    colored_print("\n=== 摄像头检查 ===", 'blue')
    
    try:
        import cv2
        
        # 检查USB摄像头
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                colored_print(f"  ✓ USB摄像头可用 (分辨率: {frame.shape[1]}x{frame.shape[0]})", 'green')
            else:
                colored_print("  ⚠️ USB摄像头已连接但无法读取图像", 'yellow')
            cap.release()
        else:
            colored_print("  ❌ USB摄像头不可用", 'red')
        
        # 检查CSI摄像头（Jetson特有）
        if os.path.exists('/etc/nv_tegra_release'):
            gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGR ! appsink"
            cap_csi = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            if cap_csi.isOpened():
                colored_print("  ✓ CSI摄像头可用", 'green')
                cap_csi.release()
            else:
                colored_print("  ⚠️ CSI摄像头不可用（可能未连接）", 'yellow')
    
    except Exception as e:
        colored_print(f"  ❌ 摄像头检查失败: {e}", 'red')

def check_model_files():
    """检查模型文件"""
    colored_print("\n=== 模型文件检查 ===", 'blue')
    
    possible_paths = [
        "weights/best.pt",
        "../weights/best.pt",
        "./best.pt",
        "best.pt"
    ]
    
    found_model = False
    for path in possible_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            colored_print(f"  ✓ 找到模型: {path} ({size:.1f}MB)", 'green')
            found_model = True
            break
    
    if not found_model:
        colored_print("  ⚠️ 未找到YOLO模型文件 (best.pt)", 'yellow')
        colored_print("    请确保模型文件位于以下位置之一:", 'yellow')
        for path in possible_paths:
            colored_print(f"    - {path}", 'yellow')
    
    # 检查TensorRT引擎
    trt_paths = [
        "weights/best_trt.engine",
        "../weights/best_trt.engine",
        "./best_trt.engine"
    ]
    
    found_trt = False
    for path in trt_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            colored_print(f"  ✓ 找到TensorRT引擎: {path} ({size:.1f}MB)", 'green')
            found_trt = True
            break
    
    if not found_trt:
        colored_print("  ⚠️ 未找到TensorRT引擎文件", 'yellow')
        colored_print("    运行 python convert_to_tensorrt.py 来创建", 'yellow')

def generate_summary(modules):
    """生成总结报告"""
    colored_print("\n" + "="*50, 'blue')
    colored_print("环境检查总结", 'blue')
    colored_print("="*50, 'blue')
    
    # 核心组件状态
    core_components = {
        'Python环境': sys.version_info >= (3, 8),
        'PyTorch': modules['torch'] is not None,
        'OpenCV': modules['cv2'] is not None,
        'EasyOCR': modules['easyocr'] is not None,
        'CUDA支持': modules['torch'] is not None and modules['torch'].cuda.is_available() if modules['torch'] else False,
        'TensorRT': modules['tensorrt'] is not None,
    }
    
    all_good = True
    for component, status in core_components.items():
        if status:
            colored_print(f"  ✓ {component}", 'green')
        else:
            colored_print(f"  ❌ {component}", 'red')
            all_good = False
    
    colored_print("\n" + "="*50, 'blue')
    
    if all_good:
        colored_print("🎉 环境配置完整！可以运行优化代码", 'green')
        colored_print("\n建议的下一步:", 'cyan')
        colored_print("1. python convert_to_tensorrt.py  # 转换模型为TensorRT", 'cyan')
        colored_print("2. python inference4_jetson_optimized.py  # 运行优化版本", 'cyan')
    else:
        colored_print("⚠️  环境配置不完整，请按照指南完成配置", 'yellow')
        colored_print("\n建议运行:", 'cyan')
        colored_print("chmod +x install_jetson_env.sh && ./install_jetson_env.sh", 'cyan')

def main():
    """主函数"""
    colored_print("🔍 Jetson环境快速检查", 'blue')
    colored_print("="*50, 'blue')
    
    # 执行各项检查
    check_system_info()
    check_performance_mode()
    check_cuda()
    modules = check_python_packages()
    check_torch_cuda(modules['torch'])
    check_opencv_cuda(modules['cv2'])
    check_camera()
    check_model_files()
    
    # 生成总结
    generate_summary(modules)

if __name__ == "__main__":
    main() 