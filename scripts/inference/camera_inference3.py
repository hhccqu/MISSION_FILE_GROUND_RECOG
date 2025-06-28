import cv2
import numpy as np
from yolo_utils import YOLODetector  # 假设这是自定义的YOLO工具类
import easyocr

class ArrowProcessor:
    def __init__(self):
        # 初始化OCR（全局单例）
        self.reader = easyocr.Reader(['en'], gpu=False)  # 根据CUDA可用性可设为True
        
        # 红色阈值范围（HSV颜色空间）
        self.lower_red1 = np.array([0, 30, 30])
        self.lower_red2 = np.array([150, 30, 30])
        self.upper_red1 = np.array([30, 255, 255])
        self.upper_red2 = np.array([179, 255, 255])  # 修正 upper_red2 的 H 值
        
        # 形态学处理核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _preprocess_red_mask(self, image):
        """红色区域预处理管道"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        combined = cv2.bitwise_or(mask1, mask2)
        
        # 形态学处理
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return cleaned

    def _correct_rotation(self, image, angle):
        """执行旋转并验证方向"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
        
        # 方向验证（基于红色区域）
        rotated_hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
        rotated_mask1 = cv2.inRange(rotated_hsv, self.lower_red1, self.upper_red1)
        rotated_mask2 = cv2.inRange(rotated_hsv, self.lower_red2, self.upper_red2)
        rotated_mask = cv2.bitwise_or(rotated_mask1, rotated_mask2)
        
        # 比较上下半区
        top = rotated_mask[:h//2, :]
        bottom = rotated_mask[h//2:, :]
        if cv2.countNonZero(bottom) > cv2.countNonZero(top):
            rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        return rotated

    def rotate_arrow(self, crop_image):
        """核心旋转校正流程"""
        # 红色区域检测
        mask = self._preprocess_red_mask(crop_image)
        
        # 轮廓分析
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop_image  # 无轮廓时退回原图
            
        # 最大轮廓处理
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        (_, _), (w, h), angle = rect
        
        # 角度修正逻辑
        if w > h:
            angle += 90
        return self._correct_rotation(crop_image, angle)

    def ocr_recognize(self, image):
        """执行OCR识别"""
        # 预处理增强对比度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 执行OCR
        results = self.reader.readtext(enhanced, detail=0)
        return " ".join(results).upper()  # 返回合并后的大写字符串

def main():
    # 初始化检测器和处理器
    detector = YOLODetector(model_path="D:/AirmodelingTeam/CQU_Ground_Recog_Strile_YoloOcr/weights/best.pt", conf_thres=0.25)
    processor = ArrowProcessor()
    
    # 视频输入设置
    cap = cv2.VideoCapture(r"D:\AirmodelingTeam\video1.mp4")
    if not cap.isOpened():
        raise IOError("无法打开视频文件")

    while True:
        ret, frame = cap.read()

        
        if not ret:
            break

        # YOLO检测
        detections = detector.detect(frame)
        
        # 动态显示参数
        preview_height = 120
        preview_width = 120
        spacing = 10  # 图像间距
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['box'])
            
            try:
                # 扩展比例
                expand_ratio = 0.1
                
                # 计算原始检测框的宽度和高度
                width = x2 - x1
                height = y2 - y1
                
                # 计算扩展量
                expand_w = int(width * expand_ratio)
                expand_h = int(height * expand_ratio)
                
                # 计算扩展后的裁剪坐标，确保不超出图像边界
                x1_exp = max(0, x1 - expand_w)
                y1_exp = max(0, y1 - expand_h)
                x2_exp = min(frame.shape[1], x2 + expand_w)
                y2_exp = min(frame.shape[0], y2 + expand_h)
                
                # 裁剪扩展后的区域
                crop = frame[y1_exp:y2_exp, x1_exp:x2_exp].copy()
                if crop.size == 0:
                    continue
                
                # 旋转校正
                rotated = processor.rotate_arrow(crop)
                
                # OCR识别
                text = processor.ocr_recognize(rotated)
                
                # 可视化检测框和OCR结果
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 动态显示旋转校正后的图像
                preview = cv2.resize(rotated, (preview_width, preview_height))
                x_offset = 10 + i * (preview_width + spacing)
                y_offset = 10
                # 确保预览区域不超出帧边界
                if x_offset + preview_width <= frame.shape[1] and y_offset + preview_height <= frame.shape[0]:
                    frame[y_offset:y_offset + preview_height, x_offset:x_offset + preview_width] = preview
                
            except Exception as e:
                print(f"处理异常: {str(e)}")
                continue

        # 显示结果
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()





















    