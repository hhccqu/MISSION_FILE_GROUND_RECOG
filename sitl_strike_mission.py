#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SITL打击任务系统 - 连接Mission Planner SITL仿真
使用发现的正确连接: tcp:127.0.0.1:5762
"""

import cv2
import numpy as np
import time
import math
from pymavlink import mavutil
from ultralytics import YOLO
import easyocr
import re

class SITLStrikeMissionSystem:
    """连接SITL仿真的打击任务系统"""
    
    def __init__(self, video_source, connection_string="tcp:127.0.0.1:5762"):
        """
        初始化SITL打击任务系统
        
        Args:
            video_source: 视频源路径
            connection_string: SITL连接字符串，默认使用发现的端口5762
        """
        # 视频源
        self.video_source = video_source
        self.cap = None
        
        # SITL连接配置 - 使用发现的正确端口
        self.connection_string = connection_string
        print(f"🔗 SITL连接字符串: {self.connection_string}")
        
        # MAVLink连接
        self.connection = None
        self.target_system = None
        self.target_component = None
        
        # YOLO模型
        self.model = None
        
        # OCR识别器
        self.ocr_reader = None
        
        # 任务状态
        self.mission_active = False
        self.targets_detected = []
        
        # GPS和姿态信息
        self.current_gps = None
        self.current_attitude = None
        
    def initialize_systems(self):
        """初始化所有系统"""
        print("🚁 初始化SITL打击任务系统...")
        
        # 1. 初始化视频源
        if not self._init_video():
            return False
            
        # 2. 连接SITL
        if not self._connect_sitl():
            return False
            
        # 3. 初始化YOLO
        if not self._init_yolo():
            return False
            
        # 4. 初始化OCR
        if not self._init_ocr():
            return False
            
        print("✅ 所有系统初始化完成")
        return True
    
    def _init_video(self):
        """初始化视频源"""
        print(f"📹 初始化视频源: {self.video_source}")
        
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                print(f"❌ 无法打开视频源: {self.video_source}")
                return False
            
            # 获取视频信息
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ 视频源已打开: {width}x{height} @ {fps:.1f}fps")
            return True
            
        except Exception as e:
            print(f"❌ 视频源初始化失败: {e}")
            return False
    
    def _connect_sitl(self):
        """连接SITL仿真"""
        print(f"🔗 连接SITL: {self.connection_string}")
        
        try:
            # 创建MAVLink连接
            self.connection = mavutil.mavlink_connection(self.connection_string)
            
            # 等待心跳包
            print("⏳ 等待SITL心跳包...")
            heartbeat = self.connection.wait_heartbeat(timeout=10)
            
            if heartbeat:
                self.target_system = self.connection.target_system
                self.target_component = self.connection.target_component
                
                print("✅ SITL连接成功!")
                print(f"   系统ID: {self.target_system}")
                print(f"   组件ID: {self.target_component}")
                print(f"   飞控类型: {heartbeat.type}")
                print(f"   自驾仪: {heartbeat.autopilot}")
                
                return True
            else:
                print("❌ 未收到SITL心跳包")
                return False
                
        except Exception as e:
            print(f"❌ SITL连接失败: {e}")
            return False
    
    def _init_yolo(self):
        """初始化YOLO模型"""
        print("🎯 初始化YOLO模型...")
        
        try:
            # 模型加载优先级
            model_paths = ['best1.pt', 'best.pt', 'yolov8n.pt', 'yolov8s.pt']
            
            for model_path in model_paths:
                try:
                    self.model = YOLO(model_path)
                    print(f"✅ YOLO模型加载成功: {model_path}")
                    return True
                except:
                    continue
            
            print("❌ 无法加载任何YOLO模型")
            return False
            
        except Exception as e:
            print(f"❌ YOLO初始化失败: {e}")
            return False
    
    def _init_ocr(self):
        """初始化OCR识别器"""
        print("🔤 初始化OCR识别器...")
        
        try:
            self.ocr_reader = easyocr.Reader(['en', 'ch_sim'])
            print("✅ OCR识别器初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ OCR初始化失败: {e}")
            return False
    
    def get_sitl_data(self):
        """获取SITL数据"""
        if not self.connection:
            return
        
        try:
            # 非阻塞接收消息
            msg = self.connection.recv_match(blocking=False, timeout=0.1)
            
            if msg:
                msg_type = msg.get_type()
                
                # 处理GPS位置信息
                if msg_type == 'GLOBAL_POSITION_INT':
                    self.current_gps = {
                        'lat': msg.lat / 1e7,
                        'lon': msg.lon / 1e7,
                        'alt': msg.alt / 1000.0,
                        'relative_alt': msg.relative_alt / 1000.0
                    }
                
                # 处理姿态信息
                elif msg_type == 'ATTITUDE':
                    self.current_attitude = {
                        'roll': math.degrees(msg.roll),
                        'pitch': math.degrees(msg.pitch),
                        'yaw': math.degrees(msg.yaw)
                    }
        
        except Exception as e:
            print(f"⚠️ 获取SITL数据失败: {e}")
    
    def _rotate_arrow(self, crop_image):
        """箭头旋转校正（简化版）"""
        if crop_image is None or crop_image.size == 0:
            return crop_image
        
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
            
            # 检测红色区域（箭头通常是红色）
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([30, 255, 255])
            lower_red2 = np.array([150, 30, 30])  
            upper_red2 = np.array([179, 255, 255])
            
            # 创建红色掩膜
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # 形态学操作清理掩膜
            kernel = np.ones((3,3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大轮廓（假设是箭头主体）
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 计算最小外接矩形
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                
                # 调整角度到合理范围
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                
                # 如果角度太小，可能不需要旋转
                if abs(angle) > 5:  # 只有角度大于5度才旋转
                    # 获取旋转矩阵
                    center = (crop_image.shape[1]//2, crop_image.shape[0]//2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    # 应用旋转
                    rotated = cv2.warpAffine(crop_image, rotation_matrix, 
                                           (crop_image.shape[1], crop_image.shape[0]))
                    return rotated
            
            # 如果没有找到明显的箭头，返回原图
            return crop_image
            
        except Exception as e:
            print(f"⚠️ 图像转正失败: {e}")
            return crop_image

    def process_frame(self, frame):
        """处理单帧图像"""
        # 目标检测
        results = self.model(frame)
        
        detected_targets = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 降低置信度阈值，与strike_mission_system.py保持一致
                    if confidence > 0.25:  # 使用更低的阈值
                        # 提取目标区域进行OCR
                        target_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        if target_roi.size > 0:
                            # 🔄 进行图像转正
                            rotated_roi = self._rotate_arrow(target_roi.copy())
                            
                            # OCR识别（使用转正后的图像）
                            ocr_results = self.ocr_reader.readtext(rotated_roi)
                            
                            # 简化OCR处理逻辑
                            ocr_text = ""
                            ocr_confidence = 0.0
                            
                            if ocr_results:
                                # 取置信度最高的结果
                                best_result = max(ocr_results, key=lambda x: x[2])
                                ocr_text = best_result[1]
                                ocr_confidence = best_result[2]
                            
                            # 提取数字（不强制要求纯数字）
                            numbers = re.findall(r'\d+', ocr_text)
                            detected_number = numbers[0] if numbers else "未识别"
                            
                            target_info = {
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence,
                                'class_id': class_id,
                                'number': detected_number,
                                'ocr_confidence': ocr_confidence,
                                'ocr_text': ocr_text,  # 添加原始OCR文本
                                'original_roi': target_roi,  # 保存原始ROI
                                'rotated_roi': rotated_roi,  # 保存转正后ROI
                                'gps': self.current_gps.copy() if self.current_gps else None,
                                'attitude': self.current_attitude.copy() if self.current_attitude else None
                            }
                            detected_targets.append(target_info)
        
        return detected_targets
    
    def draw_detections(self, frame, targets):
        """在帧上绘制检测结果"""
        # 在左上角显示检测统计
        detection_text = f"检测到目标: {len(targets)}"
        cv2.putText(frame, detection_text, (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 在右上角显示转正图像对比
        roi_display_x = frame.shape[1] - 320  # 右上角位置
        roi_display_y = 10
        roi_size = 150  # 显示ROI的大小
        
        for i, target in enumerate(targets):
            x1, y1, x2, y2 = target['bbox']
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签 - 显示更多信息
            label = f"目标{i+1}: {target['number']}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示置信度
            conf_text = f"YOLO: {target['confidence']:.2f}"
            cv2.putText(frame, conf_text, (x1, y1-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 显示OCR信息
            if target.get('ocr_text'):
                ocr_text = f"OCR: {target['ocr_text'][:10]}..."  # 限制长度
                cv2.putText(frame, ocr_text, (x1, y2+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # 🔄 显示图像转正效果（仅显示第一个目标的转正效果）
            if i == 0 and 'original_roi' in target and 'rotated_roi' in target:
                original_roi = target['original_roi']
                rotated_roi = target['rotated_roi']
                
                if original_roi.size > 0 and rotated_roi.size > 0:
                    # 调整ROI大小以适应显示
                    original_resized = cv2.resize(original_roi, (roi_size, roi_size))
                    rotated_resized = cv2.resize(rotated_roi, (roi_size, roi_size))
                    
                    # 在右上角显示原始图像
                    try:
                        frame[roi_display_y:roi_display_y+roi_size, 
                              roi_display_x:roi_display_x+roi_size] = original_resized
                        
                        # 添加标签
                        cv2.putText(frame, "原始图像", (roi_display_x, roi_display_y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # 绘制边框
                        cv2.rectangle(frame, (roi_display_x, roi_display_y), 
                                     (roi_display_x+roi_size, roi_display_y+roi_size), 
                                     (255, 255, 255), 2)
                        
                        # 在右上角显示转正后图像
                        rotated_y = roi_display_y + roi_size + 20
                        frame[rotated_y:rotated_y+roi_size, 
                              roi_display_x:roi_display_x+roi_size] = rotated_resized
                        
                        # 添加标签
                        cv2.putText(frame, "转正后图像", (roi_display_x, rotated_y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # 绘制边框
                        cv2.rectangle(frame, (roi_display_x, rotated_y), 
                                     (roi_display_x+roi_size, rotated_y+roi_size), 
                                     (0, 255, 0), 2)
                        
                        # 显示转正效果说明
                        effect_text = f"识别结果: {target['number']}"
                        cv2.putText(frame, effect_text, (roi_display_x, rotated_y+roi_size+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # 显示OCR置信度
                        ocr_conf_text = f"OCR置信度: {target['ocr_confidence']:.2f}"
                        cv2.putText(frame, ocr_conf_text, (roi_display_x, rotated_y+roi_size+40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                    except Exception as e:
                        print(f"⚠️ 显示转正效果失败: {e}")
        
        # 显示SITL状态
        status_text = []
        if self.current_gps:
            gps_text = f"GPS: {self.current_gps['lat']:.6f}, {self.current_gps['lon']:.6f}, {self.current_gps['alt']:.1f}m"
            status_text.append(gps_text)
        
        if self.current_attitude:
            att_text = f"姿态: R{self.current_attitude['roll']:.1f}° P{self.current_attitude['pitch']:.1f}° Y{self.current_attitude['yaw']:.1f}°"
            status_text.append(att_text)
        
        # 绘制状态信息
        for i, text in enumerate(status_text):
            cv2.putText(frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_mission(self):
        """运行打击任务"""
        if not self.initialize_systems():
            print("❌ 系统初始化失败，无法运行任务")
            return
        
        print("🚀 开始SITL打击任务...")
        print("⏰ 测试模式：仅处理前15秒视频")
        self.mission_active = True
        
        frame_count = 0
        target_count = 0
        start_time = time.time()  # 记录开始时间
        
        # 获取视频帧率
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        max_frames = int(fps * 15)  # 15秒对应的帧数
        print(f"📹 视频帧率: {fps:.1f}fps, 15秒最大帧数: {max_frames}")
        
        try:
            while self.mission_active:
                # 获取SITL数据
                self.get_sitl_data()
                
                # 读取视频帧
                ret, frame = self.cap.read()
                if not ret:
                    print("📹 视频播放完毕")
                    break
                
                frame_count += 1
                elapsed_time = time.time() - start_time
                
                # 检查是否超过15秒
                if elapsed_time > 15.0:
                    print(f"⏰ 已达到15秒测试时间限制，停止处理")
                    break
                
                # 处理帧（每5帧处理一次以提高性能）
                if frame_count % 5 == 0:
                    # 添加调试：显示YOLO原始检测结果
                    raw_results = self.model(frame)
                    total_detections = 0
                    for result in raw_results:
                        if result.boxes is not None:
                            total_detections += len(result.boxes)
                    
                    if total_detections > 0:
                        print(f"🔍 第{frame_count}帧(时间:{elapsed_time:.1f}s) YOLO原始检测: {total_detections} 个候选目标")
                    
                    targets = self.process_frame(frame)
                    
                    if targets:
                        target_count += len(targets)
                        print(f"🎯 第{frame_count}帧(时间:{elapsed_time:.1f}s) 最终确认: {len(targets)} 个有效目标")
                        
                        for target in targets:
                            print(f"   目标数字: {target['number']}, YOLO置信度: {target['confidence']:.2f}, OCR: {target.get('ocr_text', 'N/A')}")
                            if target['gps']:
                                print(f"   GPS位置: {target['gps']['lat']:.6f}, {target['gps']['lon']:.6f}")
                        
                        self.targets_detected.extend(targets)
                    elif total_detections > 0:
                        print(f"⚠️ 第{frame_count}帧(时间:{elapsed_time:.1f}s): YOLO检测到{total_detections}个候选目标，但过滤后无有效目标")
                    
                    # 绘制检测结果
                    frame = self.draw_detections(frame, targets)
                
                # 在图像上显示测试进度
                progress_text = f"测试进度: {elapsed_time:.1f}/15.0s ({elapsed_time/15*100:.1f}%)"
                cv2.putText(frame, progress_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 显示帧
                cv2.imshow('SITL打击任务系统', frame)
                
                # 按键控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 用户停止任务")
                    break
                elif key == ord('s'):
                    print(f"📊 当前统计: 处理{frame_count}帧, 时间{elapsed_time:.1f}s, 检测到{target_count}个目标")
                
                # 控制帧率
                time.sleep(0.03)  # 约30fps
        
        except KeyboardInterrupt:
            print("\n🛑 任务被中断")
        
        finally:
            # 显示测试结果摘要
            final_time = time.time() - start_time
            print(f"\n📊 15秒测试完成摘要:")
            print(f"   实际运行时间: {final_time:.1f}秒")
            print(f"   处理帧数: {frame_count}")
            print(f"   检测到目标总数: {target_count}")
            print(f"   平均FPS: {frame_count/final_time:.1f}")
            
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理系统资源...")
        
        if self.cap:
            self.cap.release()
        
        if self.connection:
            self.connection.close()
        
        cv2.destroyAllWindows()
        
        # 显示任务统计
        if self.targets_detected:
            print(f"\n📊 任务完成统计:")
            print(f"   总检测目标: {len(self.targets_detected)}")
            
            numbers = [t['number'] for t in self.targets_detected if t['number'] != "未识别"]
            unique_numbers = set(numbers)
            print(f"   识别出的数字: {sorted(unique_numbers)}")
            
            for num in unique_numbers:
                count = numbers.count(num)
                print(f"   数字 {num}: {count} 次")
            
            # 统计识别率
            total_targets = len(self.targets_detected)
            recognized_targets = len([t for t in self.targets_detected if t['number'] != "未识别"])
            recognition_rate = recognized_targets / total_targets * 100 if total_targets > 0 else 0
            print(f"   识别成功率: {recognition_rate:.1f}% ({recognized_targets}/{total_targets})")
        else:
            print("📊 未检测到任何目标")

def main():
    """主函数"""
    print("🚁 SITL打击任务系统启动")
    print("=" * 50)
    
    # 视频源路径
    video_source = "D:/AirmodelingTeam/MISSION_FILE_GROUND_RECOG/video2.mp4"
    
    # 创建任务系统（使用发现的正确连接）
    mission_system = SITLStrikeMissionSystem(
        video_source=video_source,
        connection_string="tcp:127.0.0.1:5762"
    )
    
    # 运行任务
    mission_system.run_mission()

if __name__ == "__main__":
    main() 