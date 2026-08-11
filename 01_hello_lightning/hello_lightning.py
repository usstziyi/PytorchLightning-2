"""
Lesson 1: 初识 PyTorch Lightning —— 最小可运行示例
====================================================

学习目标：
  1. 理解 Lightning 的核心理念：把"研究代码"(模型+训练循环) 与"工程代码"(训练器)
     解耦。
  2. 认识 LightningModule 与 Trainer 两个核心类。
  3. 用 7 行核心代码跑通一个 MNIST 分类任务。

对应官方文档：
  - 15 分钟入门: https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html
  - GitHub 仓库: https://github.com/Lightning-AI/pytorch-lightning

运行方式（在项目根目录）：
  uv run python 01_hello_lightning/hello_lightning.py
"""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from lightning.pytorch.profilers import PyTorchProfiler


# ---------------------------------------------------------------
# 1. 定义一个 LightningModule（本质还是一个 nn.Module，但多了"系统"级逻辑）
#    我们把 模型、训练step、验证step、优化器配置 全部放在一起。
# ---------------------------------------------------------------
class HelloLightning(L.LightningModule):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        # 保存超参数，便于 checkpoint 恢复与复现。
        # checkpoint 里包含两部分：
        # 1. hparams ： hidden_size=128 （用于重建模型）
        # 2. state_dict ： 各层（flatten/fc1/relu/fc2）的权重/偏置值（用于恢复训练状态）
        self.save_hyperparameters()

        # 提供一个示例输入，供 ModelSummary 统计每层的 FLOPs（不设置则为 0）
        self.example_input_array = torch.zeros(1, 1, 28, 28)

        # 一个简单的两层全连接网络（拆开定义，每层单独一个模块）
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    # forward 只负责"推理/预测"，不包含训练逻辑
    def forward(self, x):
        x = self.flatten(x)  # [batch, 784]
        x = self.fc1(x)      # [batch, 128]
        x = self.relu(x)     # [batch, 128]
        x = self.fc2(x)      # [batch, 10]
        return x

    # 训练循环：每个 batch 调用一次，返回 loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # self.log 用于记录指标，默认写到 TensorBoard
        # prog_bar=True 把指标 train_loss 显示在终端/进度条上，训练时你能实时看到它。
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # 验证循环：每个 batch 调用一次
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    # 测试循环：每个 batch 调用一次
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_acc", acc, prog_bar=True)

    # 优化器配置
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ---------------------------------------------------------------
# 2. 准备数据（这里为了简单，直接在脚本里写，Lesson 4 会引入 DataModule）
# ---------------------------------------------------------------
def prepare_data(batch_size: int = 64):
    transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,)),
        ]  
    )
    # 第一次运行会自动下载 MNIST 到 ./data 目录
    full = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train, val = random_split(full, [55_000, 5_000])
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=2)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------
# 3. 主流程：模型 + Trainer 就绪后，训练器自动处理 训练/验证/保存 等工程细节
# ---------------------------------------------------------------
def main():
    # 数据
    train_loader, val_loader, test_loader = prepare_data()

    # 会计
    model = HelloLightning(hidden_size=128)

    # 经理
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="mps",  # Apple Silicon 的 MPS 设备
        devices="auto", # 数量：自动检测
        log_every_n_steps=10,
        enable_progress_bar=True,
        # profiler=PyTorchProfiler(),   # ← 开启 PyTorchProfiler（统计 FLOPs 等）
    )

    # 训练 + 验证
    trainer.fit(model, train_loader, val_loader)

    # 测试
    # trainer.test(model, test_loader)


if __name__ == "__main__":
    main()