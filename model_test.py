
import torch
import torch.nn as nn
import torch.optim as optim

# 示例数据
inputs = torch.tensor([[1.0], [2.0], [3.0]])
targets = torch.tensor([[2.0], [4.0], [6.0]])

# 简单线性回归模型
model = nn.Linear(1, 1) # PyTorch 的一个线性层,这里表示该层接受一个包含 1 个特征的输入，并将其变换成一个包含 1 个特征的输出。

# 均方误差损失函数
criterion = nn.MSELoss()

# 优化器
#optim.SGD 是 PyTorch 中用于实现随机梯度下降（Stochastic Gradient Descent, SGD）优化算法的类
### 关键参数
# - `params`: 要优化的参数，通常是通过 `model.parameters()` 传递给优化器。
# - `lr`（学习率）：控制参数更新的步长。值通常设为一个小的正数。
# - `momentum`: （可选）用于加速 SGD 在相关方向上的收敛，并减少震荡。默认为 0。
# - `dampening`: （可选）冲量的抑制因子。默认为 0。
# - `weight_decay`: （可选）权重衰减（L2 正则化系数）。默认为 0。
# - `nesterov`: （可选）布尔值，表示是否使用 Nesterov 动量。默认为 `False`。
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练过程
for epoch in range(100):
    # 前向传播
    outputs = model(inputs)
    # 计算损失
    loss = criterion(outputs, targets)
    # 梯度清零(在每次参数更新前都必须调用 `optimizer.zero_grad()`，否则前面的梯度会累计)
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 参数更新优化
    optimizer.step()

    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')