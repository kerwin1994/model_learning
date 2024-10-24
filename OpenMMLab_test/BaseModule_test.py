import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from mmcv.runner import BaseModule, EpochBasedRunner

# 自定义模型
class MyModel(BaseModule):
    def __init__(self, init_cfg=None):
        super(MyModel, self).__init__(init_cfg)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 3, 32, 32)
        self.labels = torch.randint(0, 10, (100,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# 简单训练循环
def train():
    # 创建模型
    model = MyModel()

    # 数据加载器
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 定义优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 创建 runner
    runner = EpochBasedRunner(
        model=model,
        optimizer=optimizer,
        work_dir='./work_dir',
        logger=print
    )

    runner.register_training_hooks(lr_config=None, optimizer_config=None, checkpoint_config=None, log_config=None)

    # 训练函数
    def batch_processor(model, data, train_mode):
        inputs, targets = data
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return {'loss': loss}

    # 开始训练
    runner.run([dataloader], [('train', 1)], workflow=[('train', 1)], max_epochs=10)

# 执行训练
if __name__ == '__main__':
    train()