"""
Lesson 3: Trainer 详解
=======================

学习目标：
  1. 掌握 Trainer 的常用参数：accelerator / devices / max_epochs / precision /
     gradient_clip_val / accumulate_grad_batches / limit_batches / overfit_batches。
  2. 理解 Trainer 在"工程层面"自动处理了哪些事情。
  3. 学会设置可复现性 seed。

对应官方文档：
  - Trainer: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
"""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy


class TrainerDemoModel(L.LightningModule):
    def __init__(self, in_features: int = 8, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        # 这两行代码创建了两个 torchmetrics 的 准确率指标对象 ：
        # 使用 torchmetrics 的标准指标（比手写更规范、可自动聚合）
        # multiclass:声明这是 多分类 任务
        # binary （二分类）、 multilabel （多标签）
        # 支持多设备同步 ：在分布式训练时能自动处理各设备间的指标同步。
        # metric 的定位就是"数据集/epoch 级别的统计量"，不是"单 step 观测值"。
        self.train_acc = Accuracy(task="multiclass", num_classes=3)
        self.val_acc = Accuracy(task="multiclass", num_classes=3)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # 对 metric 对象，self.log 需要 *_metric 后缀或直接传聚合值
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def make_data():
    torch.manual_seed(0)
    x = torch.randn(1500, 8)
    w = torch.randn(8, 3)
    y = (x @ w).argmax(dim=1)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=64, shuffle=True)


def main():
    loader = make_data()

    # 用 overfit_batches 快速验证代码能否跑通（只在一个 batch 上反复训练）
    # 调试利器： 只用 1 个 batch 的数据反复训练 ，用来验证"代码没写错"。
    trainer_overfit = L.Trainer(
        accelerator="mps",
        devices="auto",
        max_epochs=3,
        overfit_batches=1,  # 调优常用：先确认能过拟合
        log_every_n_steps=5,
    )
    trainer_overfit.fit(TrainerDemoModel(), loader)
    print(">>> overfit_batches=1 验证通过（能在一个 batch 上过拟合即说明代码正确）\n")

    # 正式训练：展示多种工程参数
    # 精度选择：CUDA 用 fp16，MPS 用 bf16，其它回退到 32
    if torch.cuda.is_available():
        precision = "16-mixed"
    elif torch.backends.mps.is_available():
        precision = "bf16-mixed"
    else:
        precision = "32-true"


    trainer = L.Trainer(
        accelerator="mps",
        devices="auto",
        max_epochs=5,
        gradient_clip_val=1.0,        # 梯度裁剪
        accumulate_grad_batches=2,    # 累积梯度，等效 batch×2,12 个 batch 实际只做 6 次优化器更新
        precision=precision,          # 混合精度（CPU 上自动回退到 32）
        limit_train_batches=0.5,      # 每 epoch 只用 50% 训练数据（演示用）,快速跑通，缩短单 epoch 时间（本文件就是这么用的）
        log_every_n_steps=10,         # 12 个 batch 里只有 step 10 会 flush 一次 step 级日志
        enable_checkpointing=True,    # 开启模型检查点保存
    )
    trainer.fit(TrainerDemoModel(), loader)


if __name__ == "__main__":
    main()