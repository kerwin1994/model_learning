# `nn.Linear` 是 PyTorch 中用于实现全连接层（线性层）的模块。它对输入数据应用线性变换。

### 公式
# 给定输入 (\mathbf{x})，`nn.Linear` 计算：
# [
# \mathbf{y} = \mathbf{x} \mathbf{W}^\top + \mathbf{b}
# ]
# 其中：
# - (\mathbf{x}) 是输入向量。
# - (\mathbf{W}) 是权重矩阵。
# - (\mathbf{b}) 是偏置向量。
# - (\mathbf{y}) 是输出向量。
# ### 参数
# - in_features: 输入特征的数量。
# - out_features: 输出特征的数量。
# - bias: 是否包含偏置项，默认是 `True`。

import torch
import torch.nn as nn

# 创建一个线性层
linear = nn.Linear(in_features=10, out_features=5)
# 示例输入
input_data = torch.randn(3, 10)  # 3个样本，每个样本10个特征
# 前向传播
output = linear(input_data)
print(output.shape)  # 输出: torch.Size([3, 5])

### 应用场景
# - 神经网络层: 常用于构建神经网络的隐藏层或输出层。
# - 特征变换: 将输入特征映射到另一维度的特征空间。
# `nn.Linear` 模块是构建深度学习模型的基础组件，适用于需要执行线性变换的任务。
