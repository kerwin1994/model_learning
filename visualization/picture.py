import cv2
import matplotlib.pyplot as plt
import subprocess
import json
from datetime import datetime, timedelta

def get_video_start_time(video_path):
    # 使用 ffprobe 获取视频信息
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = json.loads(result.stdout)
    # 提取创建时间
    creation_time = info['format']['tags'].get('creation_time', None)
    return creation_time


def get_frame_by_absolute_timestamp(video_path, absolute_timestamp):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频的开始时间和目标绝对时间戳
    video_start = get_video_start_time(video_path)
    target_time = absolute_timestamp
    # 计算相对时间差
    time_diff = target_time - video_start
    print(target_time)
    print(video_start)
    
    seconds_diff = time_diff.total_seconds()
    # 计算目标帧号
    frame_number = int(seconds_diff * fps)
    return frame_number


#
def display_frame(video_path, frame_number):
    # 创建视频捕获对象
    cap = cv2.VideoCapture(video_path)
    # 检查是否成功打开视频
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 设置要提取的帧号
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return

    # 将 BGR 转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 使用 Matplotlib 显示图像
    plt.imshow(frame_rgb)
    plt.axis('off')  # 关闭坐标轴
    plt.title(f'Frame {frame_number}')
    plt.show()
    # 释放视频捕获对象
    cap.release()

# 使用示例
video_path = '/home/kerwin/Tool/LisvitX_v0.78.14_for_20.04/LW433B121N10012981729589162404_20241022172602/FEE_LW433B121N1001298_256_0_1729589162404_2_0_v1.0.1_20240919160802.mp4'  # 替换为你的视频文件路径
absolute_timestamp = '1729589153206'  # 绝对目标时间戳
video_frame=get_frame_by_absolute_timestamp(video_path,  absolute_timestamp)

display_frame(video_path, video_frame)  # 显示第10帧