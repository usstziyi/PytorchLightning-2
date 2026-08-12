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
from torch.utils.data import DataLoader, TensorDataset, random_split
# PyTorch 的评估指标库
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from lightning.pytorch.callbacks import RichProgressBar


class TrainerDemoModel(L.LightningModule):
    def __init__(self, in_features: int = 8, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        # MetricCollection 把多个指标打包成一个对象，一次 update、一次 log_dict
        # 就能同时统计 Accuracy / Precision / Recall / F1Score。
        # task="multiclass" 声明这是多分类任务（还有 binary 二分类、multilabel 多标签），
        # num_classes=3 与网络输出维度一致；多设备同步、epoch 级聚合由 torchmetrics 自动处理。
        metric_kwargs = {"task": "multiclass", "num_classes": 3}
        self.train_metrics = MetricCollection(
            {
                "train_acc": Accuracy(**metric_kwargs),
                "train_prec": Precision(**metric_kwargs),
                "train_rec": Recall(**metric_kwargs),
                "train_f1": F1Score(**metric_kwargs),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "val_acc": Accuracy(**metric_kwargs),
                "val_prec": Precision(**metric_kwargs),
                "val_rec": Recall(**metric_kwargs),
                "val_f1": F1Score(**metric_kwargs),
            }
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # 一次调用同时更新集合里的 4 个指标
        self.train_metrics(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        # log_dict 会把集合里的每个指标都注册给 Trainer，epoch 结束时自动 compute
        self.log_dict(self.train_metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_metrics(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        # 验证阶段 on_epoch 默认开启，每个 epoch 结束后输出全部指标
        self.log_dict(self.val_metrics, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def make_data():
    torch.manual_seed(0)
    x = torch.randn(1500, 8)
    w = torch.randn(8, 3)
    y = (x @ w).argmax(dim=1)
    ds = TensorDataset(x, y)
    # 按 8:2 拆分训练集和验证集（random_split 在固定 seed 下可复现）
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    return train_loader, val_loader


def main():
    train_loader, val_loader = make_data()

    # 用 overfit_batches 快速验证代码能否跑通（只在一个 batch 上反复训练）
    # 调试利器： 只用 1 个 batch 的数据反复训练 ，用来验证"代码没写错"。
    trainer_overfit = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=3,
        overfit_batches=1,  # 调优常用：先确认能过拟合
        log_every_n_steps=5,
    )
    trainer_overfit.fit(TrainerDemoModel(), train_loader)
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
        accelerator="auto",
        devices="auto",
        max_epochs=20,
        gradient_clip_val=1.0,        # 梯度裁剪
        accumulate_grad_batches=2,    # 累积梯度，等效 batch×2，优化器更新次数减半
        precision=precision,          # 混合精度（CPU 上自动回退到 32）
        limit_train_batches=0.5,      # 每 epoch 只用 50% 训练数据（演示用）,快速跑通，缩短单 epoch 时间（本文件就是这么用的）
        log_every_n_steps=10,         # 每 epoch 的 batch 里只有 step 10 会 flush 一次 step 级日志
        enable_checkpointing=True,    # 开启模型检查点保存
        callbacks=[RichProgressBar(leave=True)],
    )
    trainer.fit(TrainerDemoModel(), train_loader, val_loader)


if __name__ == "__main__":
    main()