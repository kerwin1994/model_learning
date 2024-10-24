# `torch.cat` 中的 `dim` 参数用于指定在何种维度上连接张量。

# ### 如何使用 `dim`
# - `dim=0`: 在第一个维度（通常是行）上连接，意味着增加样本数量。
# - `dim=1`: 在第二个维度（通常是列）上连接，意味着增加特征数量。
# - `dim=-1`: 指的是在最后一个维度上连接张量

import torch

# 创建两个示例张量
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# 在第0维（行）上连接
result1 = torch.cat((tensor1, tensor2), dim=0)

print("连接在第0维 (行):\n", result1)

# 输出:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

# 在第1维（列）上连接
result2 = torch.cat((tensor1, tensor2), dim=1)
print("连接在第1维 (列):\n", result2)


# 输出:
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])

# 最后一个维度上连接张量
result2 = torch.cat((tensor1, tensor2), dim=-1)
print("最后一个维度上连接张量:\n", result2)


# 输出:
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])


### 重要注意
# - 被连接的张量在指定维度以外的维度上，尺寸必须相同。
# - `dim` 参数根据具体需求调整，确保连接后的张量符合预期的形状。
# `torch.cat` 中的 `dim` 参数提供了灵活的数据合并方式，适用于各种张量组合场景。
