# `nn.Conv2d` 是 PyTorch 中用于实现二维卷积操作的模块，是卷积神经网络（CNN）的核心部分。

# ### 参数说明
# in_channels: 输入图像的通道数，例如，RGB图像为3通道。
# out_channels: 卷积核的数量，即输出特征图的通道数。
# kernel_size: 卷积核的尺寸，可以是单个整数或元组。
# stride: 卷积的步长，控制步幅的大小。默认值为1。
# padding: 在输入张量周围添加零填充，控制输出尺寸。
# dilation: 卷积核元素间的间距，默认为1。
# groups: 控制输入和输出之间的连接，可以用于深度可分离卷积。
# bias: 如果设置为 `True`，则向输出添加偏置。
### 示例用法
import torch
import torch.nn as nn

# 创建一个 2D 卷积层
conv_layer = nn.Conv2d(
    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
)

# 输入数据 (批大小, 通道数, 高度, 宽度)
input_data = torch.randn(1, 3, 32, 32)  # 例如一个32x32的RGB图像

# 前向传播
output = conv_layer(input_data)

print(output.shape)  # 输出: torch.Size([1, 16, 32, 32])

### 应用场景
# 图像分类: 从图片中提取空间特征用于分类。
# 目标检测: 辨识和定位图片中的物体。
# 图像分割: 将图像划分成语义上有意义的区域。
# `nn.Conv2d` 是构建深度学习模型时的基础组件，用于提取和处理图像中重要的特征。
