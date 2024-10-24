import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


# 自定义数据集
class RandomDataset(Dataset):
    def __init__(self, num_samples=100):
        # 初始化数据
        self.data = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, 10, (num_samples,))
        print("data:", self.data)
        print("labels:", self.labels)

    def __len__(self):
        # 返回数据集的大小
        print("data len:", len(self.data))
        return len(self.data)

    def __getitem__(self, index):
        # 返回第 index 个样本和对应的标签
        return self.data[index], self.labels[index]


# 训练函数
def train(model, dataloader, optimizer, criterion, num_epochs=5):
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
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
            # log打印
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )


# 主程序
if __name__ == "__main__":
    # 初始化模型、数据集、数据加载器、损失函数和优化器
    model = SimpleCNN()
    dataset = RandomDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 开始训练
    train(model, dataloader, optimizer, criterion)

### 代码说明
# 模型定义: 使用 `nn.Module` 创建了一个简单的卷积神经网络。
# 数据集类: `RandomDataset` 生成随机图片和分类标签。
# 训练循环:
# 进行前向传播，计算损失。
# 执行反向传播，更新模型参数。
# 每10步打印一次损失。
