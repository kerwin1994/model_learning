# `nn.Dropout` 是 PyTorch 中用于正则化的层，通过在训练期间随机地将一些神经元的输出置零，减少过拟合现象。这种技术可以防止模型对特定数据特征的过于依赖，从而提高其泛化能力。

### 工作原理
# - 丢弃率 `p`: 每个神经元在训练过程中都有概率 ( p ) 被随机丢弃。
# - 缩放: 在测试阶段，神经元的输出会被按比例缩放，以确保训练和测试时的输出期望值保持一致。
### 参数
# - p: 丢弃的概率，常用值为 0.5。
# - inplace: 如果设置为 `True`，则将在原地进行操作，节省内存。

import torch
import torch.nn as nn

# 创建 Dropout 层，丢弃概率为 0.5
dropout = nn.Dropout(p=0.5)
# 示例输入张量
input_data = torch.randn(5, 3)
# 应用 Dropout
output = dropout(input_data)
print(input_data)
print(output)
### 使用场景
# - 防止过拟合: 使模型不会过分依赖某些神经元的输出。
# - 训练深度网络: 提高模型的泛化性能，通常用于隐藏层的输出。
### 注意事项
# - 仅在训练时启用: 在评估或测试模式下，Dropout 会自动停止工作，实现输出一致性。
# - 不做输入层的 Dropout: 通常不在输入层上使用，因为这会导致信息损失。
# `nn.Dropout` 是在PyTorch中处理过拟合问题的常用工具，是提升模型泛化能力的有效手段。
