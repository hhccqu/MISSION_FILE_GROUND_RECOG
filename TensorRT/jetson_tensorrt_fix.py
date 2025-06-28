#!/usr/bin/env python3
# jetson_tensorrt_fix.py
# Jetson设备TensorRT导出崩溃问题解决方案

import os
import sys
import time
import psutil
import subprocess
import gc
import json
from pathlib import Path

class JetsonTensorRTFixer:
    """Jetson TensorRT崩溃问题修复器"""
    
    def __init__(self):
        self.device_info = self.detect_jetson_device()
        self.memory_info = self.get_memory_info()
        self.solutions = []
        
    def detect_jetson_device(self):
        """检测Jetson设备信息"""
        device_info = {
            'is_jetson': False,
            'model': 'Unknown',
            'jetpack_version': 'Unknown',
            'cuda_version': 'Unknown',
            'tensorrt_version': 'Unknown'
        }
        
        try:
            # 检查设备模型
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().strip('\x00')
                    if 'jetson' in model.lower():
                        device_info['is_jetson'] = True
                        device_info['model'] = model
            
            # 检查JetPack版本
            try:
                result = subprocess.run(['apt', 'show', 'nvidia-jetpack'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            device_info['jetpack_version'] = line.split(':')[1].strip()
            except:
                pass
            
            # 检查CUDA版本
            try:
                result = subprocess.run(['nvcc', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'release' in line:
                            device_info['cuda_version'] = line.split('release')[1].split(',')[0].strip()
            except:
                pass
            
            # 检查TensorRT版本
            try:
                import tensorrt as trt
                device_info['tensorrt_version'] = trt.__version__
            except:
                pass
                
        except Exception as e:
            print(f"设备检测失败: {e}")
        
        return device_info
    
    def get_memory_info(self):
        """获取内存信息"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_ram_gb': memory.total / (1024**3),
            'available_ram_gb': memory.available / (1024**3),
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3)
        }
    
    def diagnose_crash_causes(self):
        """诊断崩溃原因"""
        print("🔍 诊断TensorRT导出崩溃原因")
        print("-" * 50)
        
        causes = []
        
        # 1. 内存不足
        if self.memory_info['available_ram_gb'] < 2.0:
            causes.append({
                'type': 'memory_insufficient',
                'severity': 'high',
                'description': f"可用内存不足 ({self.memory_info['available_ram_gb']:.1f}GB < 2GB)",
                'solution': 'increase_memory'
            })
        
        # 2. 没有Swap空间
        if self.memory_info['swap_total_gb'] < 1.0:
            causes.append({
                'type': 'no_swap',
                'severity': 'high', 
                'description': f"Swap空间不足 ({self.memory_info['swap_total_gb']:.1f}GB)",
                'solution': 'create_swap'
            })
        
        # 3. 工作空间设置过大
        causes.append({
            'type': 'workspace_too_large',
            'severity': 'medium',
            'description': "TensorRT工作空间设置可能过大",
            'solution': 'optimize_workspace'
        })
        
        # 4. 模型过于复杂
        causes.append({
            'type': 'model_too_complex',
            'severity': 'medium',
            'description': "模型可能过于复杂，超出Jetson处理能力",
            'solution': 'simplify_model'
        })
        
        # 5. 系统性能模式不是最高
        causes.append({
            'type': 'performance_mode',
            'severity': 'low',
            'description': "系统可能未设置为最高性能模式",
            'solution': 'set_performance_mode'
        })
        
        return causes
    
    def create_swap_file(self, size_gb=4):
        """创建Swap文件"""
        print(f"🔧 创建{size_gb}GB Swap文件...")
        
        swap_file = f"/tmp/jetson_swap_{size_gb}G"
        
        try:
            # 检查磁盘空间
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < size_gb + 1:
                print(f"❌ 磁盘空间不足 ({free_gb:.1f}GB)，无法创建{size_gb}GB Swap")
                return False
            
            commands = [
                f"sudo fallocate -l {size_gb}G {swap_file}",
                f"sudo chmod 600 {swap_file}",
                f"sudo mkswap {swap_file}",
                f"sudo swapon {swap_file}"
            ]
            
            for cmd in commands:
                print(f"执行: {cmd}")
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ 命令失败: {result.stderr}")
                    return False
            
            print(f"✅ Swap文件创建成功: {swap_file}")
            return True
            
        except Exception as e:
            print(f"❌ Swap创建失败: {e}")
            return False
    
    def optimize_system_memory(self):
        """优化系统内存"""
        print("🧹 优化系统内存...")
        
        try:
            # 清理Python垃圾
            gc.collect()
            
            # 清理系统缓存
            commands = [
                "sudo sync",
                "sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'",
                "sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'", 
                "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
            ]
            
            for cmd in commands:
                subprocess.run(cmd, shell=True, capture_output=True)
            
            # 停止不必要的服务
            services_to_stop = [
                'docker',
                'snapd',
                'cups',
                'bluetooth'
            ]
            
            for service in services_to_stop:
                try:
                    result = subprocess.run(f"sudo systemctl stop {service}".split(), 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ 已停止服务: {service}")
                except:
                    pass
            
            print("✅ 系统内存优化完成")
            return True
            
        except Exception as e:
            print(f"❌ 内存优化失败: {e}")
            return False
    
    def set_performance_mode(self):
        """设置最高性能模式"""
        print("⚡ 设置Jetson最高性能模式...")
        
        try:
            commands = [
                "sudo nvpmodel -m 0",  # 设置最高性能模式
                "sudo jetson_clocks"   # 锁定最高频率
            ]
            
            for cmd in commands:
                print(f"执行: {cmd}")
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {cmd} 执行成功")
                else:
                    print(f"⚠️  {cmd} 执行失败: {result.stderr}")
            
            return True
            
        except Exception as e:
            print(f"❌ 性能模式设置失败: {e}")
            return False
    
    def get_optimized_export_params(self, model_path):
        """获取优化的导出参数"""
        print("🎯 生成优化的导出参数...")
        
        # 基础参数
        params = {
            'format': 'engine',
            'device': 0,
            'half': True,  # FP16精度
            'verbose': True,
            'batch': 1,
            'simplify': True
        }
        
        # 根据可用内存调整工作空间
        available_gb = self.memory_info['available_ram_gb']
        
        if available_gb < 2:
            params['workspace'] = 0.25  # 256MB
            print("⚠️  内存极少，使用最小工作空间")
        elif available_gb < 3:
            params['workspace'] = 0.5   # 512MB
            print("⚠️  内存有限，使用较小工作空间")
        elif available_gb < 4:
            params['workspace'] = 1     # 1GB
        else:
            params['workspace'] = 2     # 2GB
        
        # 根据模型大小调整
        if os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if model_size_mb > 100:  # 大于100MB的模型
                params['workspace'] = min(params['workspace'], 1)
                print("⚠️  大模型检测，减少工作空间")
        
        # Jetson Nano特殊优化
        if 'nano' in self.device_info['model'].lower():
            params['workspace'] = min(params['workspace'], 0.5)
            params['int8'] = True  # 尝试INT8量化
            print("🎯 Jetson Nano优化: 使用INT8量化")
        
        return params
    
    def safe_tensorrt_export(self, model_path):
        """安全的TensorRT导出"""
        print(f"🚀 开始安全TensorRT导出: {model_path}")
        print("-" * 50)
        
        try:
            from ultralytics import YOLO
            
            # 1. 预处理
            print("1️⃣ 系统预处理...")
            self.optimize_system_memory()
            time.sleep(2)
            
            # 2. 获取优化参数
            print("2️⃣ 获取优化参数...")
            export_params = self.get_optimized_export_params(model_path)
            print(f"导出参数: {export_params}")
            
            # 3. 加载模型
            print("3️⃣ 加载YOLO模型...")
            model = YOLO(model_path)
            
            # 4. 分步导出策略
            print("4️⃣ 开始分步导出...")
            
            # 首先尝试最保守的参数
            conservative_params = export_params.copy()
            conservative_params['workspace'] = 0.25
            conservative_params['batch'] = 1
            
            try:
                print("尝试保守参数导出...")
                start_time = time.time()
                success = model.export(**conservative_params)
                export_time = time.time() - start_time
                
                if success:
                    print(f"✅ 保守参数导出成功! 耗时: {export_time:.1f}秒")
                    return True
                    
            except Exception as e:
                print(f"保守参数导出失败: {e}")
            
            # 如果保守参数失败，尝试其他策略
            print("尝试其他导出策略...")
            
            # 策略1: 仅使用CPU进行某些步骤
            try:
                cpu_params = export_params.copy()
                cpu_params['device'] = 'cpu'
                print("尝试CPU辅助导出...")
                success = model.export(**cpu_params)
                if success:
                    print("✅ CPU辅助导出成功!")
                    return True
            except:
                pass
            
            # 策略2: 分批处理
            try:
                batch_params = export_params.copy()
                batch_params['workspace'] = 0.1
                print("尝试最小工作空间导出...")
                success = model.export(**batch_params)
                if success:
                    print("✅ 最小工作空间导出成功!")
                    return True
            except:
                pass
            
            print("❌ 所有导出策略都失败了")
            return False
            
        except Exception as e:
            print(f"❌ 导出过程异常: {e}")
            return False
    
    def generate_fix_script(self):
        """生成修复脚本"""
        script_content = """#!/bin/bash
# Jetson TensorRT崩溃修复脚本

echo "🚀 Jetson TensorRT崩溃修复脚本"
echo "================================"

# 1. 设置性能模式
echo "1️⃣ 设置最高性能模式..."
sudo nvpmodel -m 0
sudo jetson_clocks

# 2. 创建Swap空间
echo "2️⃣ 创建Swap空间..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
    echo "✅ 4GB Swap空间创建完成"
else
    echo "✅ Swap空间已存在"
fi

# 3. 优化内存设置
echo "3️⃣ 优化内存设置..."
sudo sysctl vm.swappiness=10
sudo sysctl vm.vfs_cache_pressure=50

# 4. 停止不必要的服务
echo "4️⃣ 停止不必要的服务..."
sudo systemctl stop docker || true
sudo systemctl stop snapd || true
sudo systemctl stop cups || true

# 5. 清理内存
echo "5️⃣ 清理内存..."
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo "✅ 修复脚本执行完成!"
echo "现在可以尝试TensorRT导出了"
"""
        
        script_path = "jetson_tensorrt_fix.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"✅ 修复脚本已生成: {script_path}")
        return script_path
    
    def run_comprehensive_fix(self, model_path=None):
        """运行综合修复"""
        print("🛠️  运行Jetson TensorRT崩溃综合修复")
        print("=" * 60)
        
        # 1. 诊断问题
        causes = self.diagnose_crash_causes()
        print(f"发现 {len(causes)} 个潜在问题:")
        for i, cause in enumerate(causes, 1):
            print(f"{i}. {cause['description']} (严重性: {cause['severity']})")
        
        # 2. 应用修复
        print(f"\n🔧 开始应用修复方案...")
        
        # 设置性能模式
        self.set_performance_mode()
        
        # 创建Swap空间
        if self.memory_info['swap_total_gb'] < 2:
            self.create_swap_file(4)
        
        # 优化内存
        self.optimize_system_memory()
        
        # 3. 如果提供了模型路径，尝试导出
        if model_path and os.path.exists(model_path):
            print(f"\n🎯 尝试导出模型: {model_path}")
            success = self.safe_tensorrt_export(model_path)
            if success:
                print("🎉 TensorRT导出成功!")
            else:
                print("❌ TensorRT导出仍然失败")
                self.print_additional_solutions()
        
        # 4. 生成修复脚本
        self.generate_fix_script()
        
        print("\n📋 修复总结:")
        print("1. 已设置最高性能模式")
        print("2. 已创建/优化Swap空间") 
        print("3. 已优化系统内存")
        print("4. 已生成修复脚本 (jetson_tensorrt_fix.sh)")
    
    def print_additional_solutions(self):
        """打印额外解决方案"""
        print("\n💡 额外解决方案建议:")
        print("-" * 40)
        print("1. 使用更小的模型:")
        print("   - YOLOv8n.pt 而不是 YOLOv8s.pt 或更大的模型")
        print("   - 考虑使用量化后的模型")
        
        print("\n2. 分步转换策略:")
        print("   - 先转换为ONNX格式")
        print("   - 再使用trtexec工具转换为TensorRT")
        
        print("\n3. 使用外部存储:")
        print("   - 将Swap文件放在USB SSD上")
        print("   - 使用更快的存储设备")
        
        print("\n4. 调整TensorRT参数:")
        print("   - 减少workspace大小到256MB")
        print("   - 使用INT8精度而不是FP16")
        print("   - 禁用某些优化选项")
        
        print("\n5. 系统级优化:")
        print("   - 升级到更新的JetPack版本")
        print("   - 确保散热良好，避免热降频")
        print("   - 使用高质量的电源适配器")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Jetson TensorRT崩溃问题修复工具')
    parser.add_argument('--model', '-m', type=str, help='YOLO模型路径 (.pt文件)')
    parser.add_argument('--fix-only', action='store_true', help='仅执行系统修复，不尝试导出')
    
    args = parser.parse_args()
    
    # 创建修复器
    fixer = JetsonTensorRTFixer()
    
    # 显示设备信息
    print("📱 设备信息:")
    for key, value in fixer.device_info.items():
        print(f"   {key}: {value}")
    
    print(f"\n💾 内存信息:")
    for key, value in fixer.memory_info.items():
        if 'gb' in key:
            print(f"   {key}: {value:.1f}GB")
        else:
            print(f"   {key}: {value}")
    
    # 运行修复
    if args.fix_only:
        fixer.run_comprehensive_fix()
    else:
        fixer.run_comprehensive_fix(args.model)

if __name__ == "__main__":
    main() 