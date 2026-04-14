import cv2
import os

video_path = r'C:\Users\WWW\Desktop\BehaviorRecognition\demo\侧布扩胸激活.mp4'  # 改为你的标准视频地址
output_dir = r'C:\Users\WWW\Desktop\BehaviorRecognition\demo\vedio clipping' #改为你的图片保存地址
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 每隔 10 帧保存一张（假设视频是30帧/秒，一秒抽3张）
    # 这里不需要 resize！保持 frame 原样！
    if frame_count % 5 == 0:
        cv2.imwrite(f"{output_dir}/frame_{saved_count:04d}.jpg", frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"抽帧完成，共提取了 {saved_count} 张原尺寸图片。")