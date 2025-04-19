import cv2
import numpy as np
import matplotlib.pyplot as plt

# 辅助函数：显示图像
def visualize_step(image, title):
    plt.figure(figsize=(6, 6))
    if len(image.shape) == 2:  # 灰度图
        plt.imshow(image, cmap='gray')
    else:  # 彩色图
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# 旋转校正主函数
def rotate_arrow_up(image_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    visualize_step(img, "步骤 1: 原始图像")

    # 2. 转换为 HSV 并检测红色区域
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 调整后的阈值，捕获浅红色
    lower_red1 = np.array([0, 50, 50])    # 放宽饱和度和明度
    upper_red1 = np.array([20, 255, 255]) # 扩展色调范围
    lower_red2 = np.array([150, 50, 50])  # 放宽饱和度和明度
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    visualize_step(mask, "步骤 2: 红色掩码")

    # 3. 形态学处理去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    visualize_step(mask, "步骤 3: 清理后的掩码")

    # 4. 查找轮廓并计算旋转角度
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到任何红色轮廓")
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. 获取最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), raw_angle = rect

    # 6. 根据宽高关系调整角度
    angle = raw_angle if w < h else raw_angle + 90

    # 7. 执行旋转
    (h_img, w_img) = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w_img, h_img), borderValue=(255, 255, 255))
    visualize_step(rotated, f"步骤 4: 旋转 {angle:.2f} 度")

    # 8. 验证箭头方向并调整
    rotated_hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    rotated_mask1 = cv2.inRange(rotated_hsv, lower_red1, upper_red1)
    rotated_mask2 = cv2.inRange(rotated_hsv, lower_red2, upper_red2)
    rotated_mask = cv2.bitwise_or(rotated_mask1, rotated_mask2)

    top_half = rotated_mask[0:h_img // 2, :]
    bottom_half = rotated_mask[h_img // 2:, :]
    top_count = cv2.countNonZero(top_half)
    bottom_count = cv2.countNonZero(bottom_half)

    if bottom_count > top_count:
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        visualize_step(rotated, "步骤 5: 翻转 180 度（箭头朝下）")

    return rotated

# 主程序
if __name__ == "__main__":
    # 替换为你的图片路径
    image_path = r"D:\AirmodelingTeam\TRAINPHOTO\TEST4.png"
    try:
        final_result = rotate_arrow_up(image_path)
        cv2.imwrite("rotated_final.png", final_result)
        print("旋转校正完成，结果已保存为 'rotated_final.png'")
    except Exception as e:
        print(f"发生错误: {e}")