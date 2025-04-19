import cv2
from yolo_utils import YOLODetector
from ocr_utils import rotate_arrow_up, ocr_recognition

def main():
    # 1. 初始化YOLO检测器
    detector = YOLODetector(model_path="weights/best.pt", conf_thres=0.25)

    # 2. 打开本地视频文件
    video_path = r"D:\AirmodelingTeam\video1.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件。请检查路径。")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕或读取失败。")
            break

        # 3. YOLO 检测
        boxes = detector.detect(frame)

        # 4. 对检测结果进行画框、旋转、OCR等
        for box_info in boxes:
            x1, y1, x2, y2 = box_info['box']
            conf = box_info['conf']

            # 在图像上绘制检测框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

            # 旋转并裁剪
            cropped = rotate_arrow_up(frame, (x1, y1, x2, y2))

            # OCR识别
            text = ocr_recognition(cropped)
            print(f"[INFO] OCR识别结果: {text}")

            # 将OCR结果显示在检测框下方
            cv2.putText(frame, text, (int(x1), int(y2)+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # 缩放旋转后的图像以显示在框旁边
            cropped_resized = cv2.resize(cropped, (100, 100))  # 将裁剪的图像缩小到 100x100
            frame[0:100, int(x2)+10:int(x2)+110] = cropped_resized  # 将图像显示在框右边

        # 显示实时画面（建议用保存或其他方式替代）
        try:
            cv2.imshow("Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error as e:
            print("[WARNING] GUI显示失败，可能当前环境不支持imshow：", e)
            # 如果需要保存图像，也可以在这里添加cv2.imwrite()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
