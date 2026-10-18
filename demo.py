from ultralytics import RTDETR
import cv2
import os

# 1. 加载预训练模型（第一次运行会自动下载 rtdetr-l.pt，需联网）
model = RTDETR('rtdetr-l.pt')

# 2. 指定图片路径（你左侧目录树里的图片名）
image_path = 'images/20110126213231.jpg'

# 3. 执行推理
results = model(image_path)

# 4. 画框并保存
for result in results:
    annotated_frame = result.plot()
    # 保存到 L4 文件夹下
    save_path = os.path.join('L4_动手实验', 'rtdetr_result.jpg')
    cv2.imwrite(save_path, annotated_frame)
    print(f"检测到 {len(result.boxes)} 个对象，结果已保存为 {save_path}")