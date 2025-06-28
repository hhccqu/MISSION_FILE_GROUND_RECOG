#!/usr/bin/env python3
# jetson_environment_test.py
# Jetson Orin Nano 环境配置验证测试脚本

import sys
import os
import subprocess
import time
import platform
import psutil
from datetime import datetime

class JetsonEnvironmentTester:
    """Jetson环境测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.failed_tests = []
        self.warnings = []
        
    def print_header(self, title):
        """打印测试标题"""
        print(f"\n{'='*60}")
        print(f"🔧 {title}")
        print(f"{'='*60}")
    
    def print_test(self, test_name, status, details=""):
        """打印测试结果"""
        if status == "PASS":
            print(f"✅ {test_name}: {status}")
        elif status == "FAIL":
            print(f"❌ {test_name}: {status}")
            self.failed_tests.append(test_name)
        elif status == "WARN":
            print(f"⚠️  {test_name}: {status}")
            self.warnings.append(test_name)
        else:
            print(f"ℹ️  {test_name}: {status}")
        
        if details:
            print(f"   详情: {details}")
        
        self.test_results[test_name] = {"status": status, "details": details}
    
    def run_command(self, command, timeout=10):
        """执行系统命令"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "命令超时"
        except Exception as e:
            return False, "", str(e)
    
    def test_system_info(self):
        """测试系统基本信息"""
        self.print_header("系统基本信息")
        
        # 检查是否为Jetson设备
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
            if 'Jetson' in model:
                self.print_test("Jetson设备检测", "PASS", model)
            else:
                self.print_test("Jetson设备检测", "FAIL", f"非Jetson设备: {model}")
        else:
            self.print_test("Jetson设备检测", "FAIL", "无法读取设备型号")
        
        # 系统信息
        self.print_test("操作系统", "INFO", f"{platform.system()} {platform.release()}")
        self.print_test("架构", "INFO", platform.machine())
        self.print_test("Python版本", "INFO", sys.version.split()[0])
        
        # 内存信息
        memory = psutil.virtual_memory()
        self.print_test("总内存", "INFO", f"{memory.total / (1024**3):.1f} GB")
        self.print_test("可用内存", "INFO", f"{memory.available / (1024**3):.1f} GB")
        
        if memory.total < 3 * 1024**3:  # 小于3GB
            self.print_test("内存容量", "WARN", "内存较少，建议创建swap空间")
        else:
            self.print_test("内存容量", "PASS", "内存充足")
    
    def test_jetpack_components(self):
        """测试JetPack组件"""
        self.print_header("JetPack组件检测")
        
        # 检查JetPack版本
        success, output, error = self.run_command("dpkg -l | grep nvidia-jetpack")
        if success and output:
            version = output.split()[2] if len(output.split()) > 2 else "未知"
            self.print_test("JetPack安装", "PASS", f"版本 {version}")
        else:
            self.print_test("JetPack安装", "FAIL", "未安装或无法检测")
        
        # 检查CUDA
        success, output, error = self.run_command("nvcc --version")
        if success:
            cuda_version = "未知"
            for line in output.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release')[1].split(',')[0].strip()
                    break
            self.print_test("CUDA安装", "PASS", f"版本 {cuda_version}")
        else:
            self.print_test("CUDA安装", "FAIL", "nvcc命令不可用")
        
        # 检查TensorRT
        success, output, error = self.run_command("dpkg -l | grep tensorrt")
        if success and output:
            # 提取TensorRT版本
            lines = output.split('\n')
            trt_version = "未知"
            for line in lines:
                if 'tensorrt' in line and not line.startswith('ii'):
                    continue
                if 'tensorrt' in line:
                    parts = line.split()
                    if len(parts) > 2:
                        trt_version = parts[2]
                        break
            self.print_test("TensorRT安装", "PASS", f"版本 {trt_version}")
        else:
            self.print_test("TensorRT安装", "FAIL", "未安装或无法检测")
    
    def test_performance_mode(self):
        """测试性能模式"""
        self.print_header("性能模式检测")
        
        # 检查nvpmodel
        success, output, error = self.run_command("sudo nvpmodel -q")
        if success:
            current_mode = "未知"
            for line in output.split('\n'):
                if 'NV Power Mode' in line:
                    current_mode = line.split(':')[1].strip()
                    break
            
            if 'MAXN' in current_mode or '15W' in current_mode or current_mode == '0':
                self.print_test("功耗模式", "PASS", f"当前模式: {current_mode}")
            else:
                self.print_test("功耗模式", "WARN", f"非最高性能模式: {current_mode}")
        else:
            self.print_test("功耗模式", "FAIL", "无法检测功耗模式")
        
        # 检查CPU频率
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                freq = int(f.read().strip()) / 1000  # 转换为MHz
            self.print_test("CPU频率", "INFO", f"{freq:.0f} MHz")
        except:
            self.print_test("CPU频率", "WARN", "无法读取CPU频率")
    
    def test_python_packages(self):
        """测试Python包"""
        self.print_header("Python包检测")
        
        required_packages = {
            'torch': 'PyTorch',
            'torchvision': 'TorchVision', 
            'tensorrt': 'TensorRT Python',
            'pycuda': 'PyCUDA',
            'cv2': 'OpenCV',
            'numpy': 'NumPy',
            'ultralytics': 'Ultralytics YOLO',
            'easyocr': 'EasyOCR',
            'matplotlib': 'Matplotlib',
            'PIL': 'Pillow'
        }
        
        for package, name in required_packages.items():
            try:
                if package == 'cv2':
                    import cv2
                    version = cv2.__version__
                elif package == 'PIL':
                    from PIL import Image
                    version = Image.__version__ if hasattr(Image, '__version__') else "已安装"
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', '已安装')
                
                self.print_test(f"{name}包", "PASS", f"版本 {version}")
            except ImportError:
                self.print_test(f"{name}包", "FAIL", "未安装")
            except Exception as e:
                self.print_test(f"{name}包", "WARN", f"导入异常: {str(e)}")
    
    def test_cuda_functionality(self):
        """测试CUDA功能"""
        self.print_header("CUDA功能测试")
        
        try:
            import torch
            
            # CUDA可用性
            if torch.cuda.is_available():
                self.print_test("CUDA可用性", "PASS", "CUDA可用")
                
                # GPU设备信息
                device_count = torch.cuda.device_count()
                self.print_test("GPU设备数量", "INFO", f"{device_count} 个GPU")
                
                if device_count > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    self.print_test("GPU型号", "INFO", gpu_name)
                    
                    # GPU内存
                    memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    self.print_test("GPU内存", "INFO", f"{memory_total:.1f} GB")
                    
                    # 简单CUDA计算测试
                    try:
                        x = torch.randn(1000, 1000).cuda()
                        y = torch.randn(1000, 1000).cuda()
                        z = torch.mm(x, y)
                        self.print_test("CUDA计算测试", "PASS", "矩阵乘法成功")
                    except Exception as e:
                        self.print_test("CUDA计算测试", "FAIL", str(e))
            else:
                self.print_test("CUDA可用性", "FAIL", "CUDA不可用")
                
        except ImportError:
            self.print_test("PyTorch导入", "FAIL", "PyTorch未安装")
    
    def test_tensorrt_functionality(self):
        """测试TensorRT功能"""
        self.print_header("TensorRT功能测试")
        
        try:
            import tensorrt as trt
            self.print_test("TensorRT导入", "PASS", f"版本 {trt.__version__}")
            
            # 创建简单的TensorRT logger
            try:
                logger = trt.Logger(trt.Logger.WARNING)
                self.print_test("TensorRT Logger", "PASS", "创建成功")
            except Exception as e:
                self.print_test("TensorRT Logger", "FAIL", str(e))
                
            # 测试PyCUDA
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit
                self.print_test("PyCUDA初始化", "PASS", "初始化成功")
            except Exception as e:
                self.print_test("PyCUDA初始化", "FAIL", str(e))
                
        except ImportError:
            self.print_test("TensorRT导入", "FAIL", "TensorRT Python包未安装")
    
    def test_opencv_functionality(self):
        """测试OpenCV功能"""
        self.print_header("OpenCV功能测试")
        
        try:
            import cv2
            self.print_test("OpenCV导入", "PASS", f"版本 {cv2.__version__}")
            
            # 检查CUDA支持
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.print_test("OpenCV CUDA支持", "PASS", "CUDA已启用")
            else:
                self.print_test("OpenCV CUDA支持", "WARN", "CUDA未启用")
            
            # 测试摄像头
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.print_test("摄像头测试", "PASS", f"分辨率 {frame.shape[1]}x{frame.shape[0]}")
                    else:
                        self.print_test("摄像头测试", "WARN", "无法读取帧")
                    cap.release()
                else:
                    self.print_test("摄像头测试", "WARN", "无法打开摄像头")
            except Exception as e:
                self.print_test("摄像头测试", "WARN", str(e))
                
        except ImportError:
            self.print_test("OpenCV导入", "FAIL", "OpenCV未安装")
    
    def test_yolo_functionality(self):
        """测试YOLO功能"""
        self.print_header("YOLO功能测试")
        
        try:
            from ultralytics import YOLO
            self.print_test("Ultralytics导入", "PASS", "导入成功")
            
            # 测试模型加载 (使用预训练模型)
            try:
                model = YOLO('yolov8n.pt')  # 使用nano模型测试
                self.print_test("YOLO模型加载", "PASS", "预训练模型加载成功")
                
                # 测试推理
                import numpy as np
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                results = model(test_image, verbose=False)
                self.print_test("YOLO推理测试", "PASS", "推理成功")
                
            except Exception as e:
                self.print_test("YOLO功能测试", "WARN", f"测试失败: {str(e)}")
                
        except ImportError:
            self.print_test("Ultralytics导入", "FAIL", "Ultralytics未安装")
    
    def test_model_files(self):
        """测试模型文件"""
        self.print_header("模型文件检测")
        
        # 可能的模型路径
        model_paths = [
            "../weights/best1.pt",
            "weights/best1.pt", 
            "../ready/weights/best1.pt",
            "best1.pt"
        ]
        
        model_found = False
        for path in model_paths:
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024*1024)  # MB
                self.print_test("YOLO模型文件", "PASS", f"找到: {path} ({file_size:.1f} MB)")
                model_found = True
                
                # 检查对应的TensorRT引擎
                engine_path = path.replace('.pt', '.engine')
                if os.path.exists(engine_path):
                    engine_size = os.path.getsize(engine_path) / (1024*1024)
                    self.print_test("TensorRT引擎", "PASS", f"找到: {engine_path} ({engine_size:.1f} MB)")
                else:
                    self.print_test("TensorRT引擎", "WARN", "未找到，需要转换")
                break
        
        if not model_found:
            self.print_test("YOLO模型文件", "FAIL", "未找到best1.pt文件")
    
    def test_jetson_stats(self):
        """测试Jetson监控工具"""
        self.print_header("Jetson监控工具测试")
        
        try:
            from jtop import jtop
            self.print_test("jtop导入", "PASS", "jetson-stats可用")
            
            # 尝试获取系统状态
            try:
                with jtop() as jetson:
                    if jetson.ok():
                        # GPU信息
                        gpu_usage = jetson.gpu.get('GR3D', {}).get('val', 0)
                        self.print_test("GPU使用率", "INFO", f"{gpu_usage}%")
                        
                        # 温度信息
                        temp = jetson.temperature.get('CPU', 0)
                        self.print_test("CPU温度", "INFO", f"{temp}°C")
                        
                        # 功耗信息
                        power = jetson.power.get('cur', 0)
                        self.print_test("当前功耗", "INFO", f"{power}W")
                        
                        self.print_test("Jetson状态监控", "PASS", "监控数据获取成功")
                    else:
                        self.print_test("Jetson状态监控", "WARN", "无法获取监控数据")
            except Exception as e:
                self.print_test("Jetson状态监控", "WARN", f"监控异常: {str(e)}")
                
        except ImportError:
            self.print_test("jtop导入", "FAIL", "jetson-stats未安装")
    
    def generate_report(self):
        """生成测试报告"""
        self.print_header("测试报告摘要")
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r['status'] == 'PASS'])
        failed_tests = len(self.failed_tests)
        warnings = len(self.warnings)
        
        print(f"📊 总测试项目: {total_tests}")
        print(f"✅ 通过: {passed_tests}")
        print(f"❌ 失败: {failed_tests}")
        print(f"⚠️  警告: {warnings}")
        
        if failed_tests > 0:
            print(f"\n❌ 失败的测试项目:")
            for test in self.failed_tests:
                print(f"   - {test}")
        
        if warnings > 0:
            print(f"\n⚠️  警告的测试项目:")
            for test in self.warnings:
                print(f"   - {test}")
        
        # 给出建议
        print(f"\n💡 建议:")
        if failed_tests == 0:
            print("   🎉 所有关键测试都通过了！您的环境配置良好。")
        else:
            print("   🔧 请根据失败的测试项目进行相应的安装和配置。")
        
        if warnings > 0:
            print("   ⚠️  请注意警告项目，这些可能影响性能或功能。")
        
        # 保存详细报告
        report_file = f"jetson_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Jetson环境测试报告\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            
            for test_name, result in self.test_results.items():
                f.write(f"{test_name}: {result['status']}\n")
                if result['details']:
                    f.write(f"  详情: {result['details']}\n")
                f.write("\n")
        
        print(f"\n📄 详细报告已保存到: {report_file}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始Jetson环境配置验证测试...")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 执行所有测试
        self.test_system_info()
        self.test_jetpack_components()
        self.test_performance_mode()
        self.test_python_packages()
        self.test_cuda_functionality()
        self.test_tensorrt_functionality()
        self.test_opencv_functionality()
        self.test_yolo_functionality()
        self.test_model_files()
        self.test_jetson_stats()
        
        # 生成报告
        self.generate_report()

def main():
    """主函数"""
    tester = JetsonEnvironmentTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 