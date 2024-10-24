import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def img_visualization(img_path,img_timestamp):
    # 读取图片
    img = mpimg.imread(img_path)
    # 使用 Matplotlib 显示图像
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.title(f'Timestamp: {img_timestamp}')
    plt.show()

# ./liviz_cli split -p /home/kerwin/Tool/test_data/Lisvit/LW433B120P10051831729644008805_20241023084008 -f psd -c f_ca_30,f_ca_120
# 使用示例
img_timestamp = "1729644006771"  # 以秒为单位的时间戳
img_name = "/f_ca_30_"+ img_timestamp+ ".jpg"
img_path = "/home/kerwin/Tool/test_data/Lisvit/LW433B120P10051831729644008805_20241023084008_JPG_Data/f_ca_30" + img_name 
print("img_timestamp:", img_timestamp)
print("img_path:", img_path)
img_visualization(img_path,img_timestamp)
