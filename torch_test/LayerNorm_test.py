# `nn.LayerNorm` 是 PyTorch 中用于执行层归一化的模块。它可以帮助改善深度神经网络的训练稳定性和训练速度。

# ### 工作原理
# - 归一化: 对每一个输入的特征进行归一化，使其均值为0，方差为1。
# - 计算范围: 归一化在每个样本的特征维度上独立进行。
# ### 与 BatchNorm 的区别
# - 作用范围: LayerNorm 在特征维度上进行归一化，而 BatchNorm 在小批量样本维度上进行归一化。
# - 适用场景: LayerNorm 更适合序列模型（如 RNN、Transformer）或稳定的小批量大小，因为不依赖于批量统计。
# ### 参数
# - normalized_shape: 要归一化的特征的形状,在 `nn.LayerNorm` 中，`normalized_shape` 参数用于指定需要归一化的特征维度。这决定了统计这些维度上的均值和方差以进行归一化。
# - 灵活性: `normalized_shape` 可以根据需要调节，适应不同输入形状。
# - 适用性: 常用于序列模型或具有复杂特征维度的输入。
# 通过合理设置 `normalized_shape`，`LayerNorm` 可以有效改善模型的训练性能和稳定性
#### 如何定义
# - 单维度张量: 如果输入是单维度张量，`normalized_shape` 就是其整个维度。
# - 多维度张量: 对于多维度张量，通常指定最后几个维度用于归一化。例如，给定输入形状为 `(batch_size, seq_length, features)`，设置 `normalized_shape=(features,)` 会在特征维度上执行归一化。
# - eps: 添加到方差以防止除零。
# - elementwise_affine: 如果为 True，该层有可学习的缩放和平移参数。

import torch
import torch.nn as nn

# 创建一个 LayerNorm 层
layer_norm = nn.LayerNorm(normalized_shape=10)
# 示例输入
input_data = torch.randn(2, 5, 10)  # (batch_size=2, seq_length=5, features=10):5个样本，每个10个特征
# 应用 LayerNorm
output = layer_norm(input_data)
print(input_data)
print(output)
### 应用场景
# - 序列模型: 比如 RNN 和 Transformer，因为它们通常有不稳定的小批量。
# - 深度学习网络: 提高训练速度和稳定性。
# `nn.LayerNorm` 是在具有复杂特征结构的模型中保持稳定性和提高训练效率的有效工具。