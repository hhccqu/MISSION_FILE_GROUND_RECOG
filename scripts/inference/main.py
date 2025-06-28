# main.py
import cv2
import math
import easyocr
import numpy as np
from ultralytics import YOLO
from pymavlink import mavutil
import time
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TargetInfo:
    text: str
    confidence: float
    position: Tuple[float, float]  # 图像中的相对位置 (0-1)
    gps_position: Tuple[float, float]  # GPS坐标 (lat, lon)

class DroneController:
    def __init__(self, connection_string: str = "udpin:localhost:14550"):
        self.connection = mavutil.mavlink_connection(connection_string)
        self.wait_for_heartbeat()
        
    def wait_for_heartbeat(self):
        print("等待飞控心跳...")
        self.connection.wait_heartbeat()
        print("已连接到飞控")
        
    def get_current_position(self) -> Tuple[float, float]:
        """获取当前GPS位置"""
        msg = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        return (msg.lat / 1e7, msg.lon / 1e7)
        
    def upload_mission(self, waypoints: List[Tuple[float, float]]):
        """上传新的航点任务"""
        # 清除现有任务
        self.connection.waypoint_clear_all_send()
        self.connection.waypoint_count_send(len(waypoints))
        
        for i, (lat, lon) in enumerate(waypoints):
            msg = self.connection.recv_match(type='MISSION_REQUEST', blocking=True)
            if msg.seq != i:
                continue
                
            self.connection.mav.send(mavutil.mavlink.MAVLink_mission_item_message(
                self.connection.target_system,
                self.connection.target_component,
                i,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 0, 0, 0, 0, 0,
                lat, lon, 50  # 高度设为50米
            ))

def compute_rotation_angle(box):
    """
    简化示例：根据检测框 box=[x1,y1,x2,y2] 计算旋转角度 (度数)。
    若你有 apex/digit_box 两类，需要根据 apex->digit_box连线算角度。
    这里仅以框自身方向为示例(同left->right)。
    """
    x1, y1, x2, y2 = box
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def rotate_and_crop(frame, box):
    """
    1. 计算 box 的旋转角 angle
    2. 以 box 中心为基点旋转整图
    3. 截取 box 区域
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    angle = compute_rotation_angle(box)

    # 构造仿射变换矩阵
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    h, w = frame.shape[:2]
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderValue=(255,255,255))

    # 在旋转后的图像中，按原 box 位置裁剪
    x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
    cropped = rotated[y1_i:y2_i, x1_i:x2_i].copy()

    return cropped

def find_median_target(targets: List[TargetInfo]) -> TargetInfo:
    """找到数值为中位数的目标"""
    if not targets:
        return None
    
    # 提取数字并排序
    numeric_targets = []
    for target in targets:
        try:
            num = float(target.text)
            numeric_targets.append((num, target))
        except ValueError:
            continue
    
    if not numeric_targets:
        return None
        
    numeric_targets.sort(key=lambda x: x[0])
    median_idx = len(numeric_targets) // 2
    return numeric_targets[median_idx][1]

def main():
    # 加载 YOLOv8 模型 (推理)
    model = YOLO("weights/best.pt")   # 你的训练好模型

    # 初始化 OCR
    reader = easyocr.Reader(['en'], gpu=True)  # Jetson上启用GPU

    # 初始化飞控连接
    drone = DroneController()

    # 打开摄像头(或RTSP)
    cap = cv2.VideoCapture(0)  # 0表示USB或CSI摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    detected_targets = []
    last_mission_update = 0
    mission_update_interval = 5  # 每5秒更新一次任务

    while True:
        ret, frame = cap.read()
        if not ret:
            print("帧获取失败")
            break

        # 获取当前GPS位置
        current_pos = drone.get_current_position()

        # 1. YOLO 推理
        results = model.predict(source=frame, conf=0.25)
        boxes = results[0].boxes
        # 如果只关心一个目标，可取最高置信度
        if len(boxes) > 0:
            for b in boxes:
                conf = float(b.conf[0].item())
                if conf > 0.5:  # 置信度阈值
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    
                    # 计算目标在图像中的相对位置
                    h, w = frame.shape[:2]
                    rel_x = (x1 + x2) / (2 * w)
                    rel_y = (y1 + y2) / (2 * h)
                    
                # 2. 旋转并裁剪
                cropped = rotate_and_crop(frame, [x1, y1, x2, y2])

                # 3. OCR 识别
                ocr_result = reader.readtext(cropped)
                    if ocr_result:
                        text = ocr_result[0][1]
                        target = TargetInfo(
                            text=text,
                            confidence=conf,
                            position=(rel_x, rel_y),
                            gps_position=current_pos
                        )
                        detected_targets.append(target)

                # 打印或保存结果
                        print(f"OCR 识别结果：{text}")

                # 画框+文字
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                        cv2.putText(frame, text, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # 定期更新任务
        current_time = time.time()
        if current_time - last_mission_update > mission_update_interval:
            median_target = find_median_target(detected_targets)
            if median_target:
                # 创建新的航点任务
                waypoints = [
                    current_pos,  # 当前位置
                    median_target.gps_position,  # 目标位置
                ]
                drone.upload_mission(waypoints)
                print(f"已更新任务，目标数字: {median_target.text}")
            
            detected_targets = []  # 清空检测列表
            last_mission_update = current_time

        # 显示画面(如不需要UI，可注释掉)
        cv2.imshow("Jetson Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
