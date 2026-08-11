"""
Lesson 5: Callbacks —— 在训练关键节点"插桩"
=============================================

学习目标：
  1. 理解 Callback 与 hook 的区别：
    - hook 写在 LightningModule 里（模型逻辑），
    - callback 是独立于模型的"工程逻辑"，可插拔、可复用。
  2. 掌握内置回调：ModelCheckpoint / EarlyStopping / LearningRateMonitor / RichProgressBar。
  3. 学会自定义一个 Callback。

对应官方文档：
  - Callbacks: https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html
"""

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy


class DemoModel(L.LightningModule):
    def __init__(self, lr=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))
        self.val_acc = Accuracy(task="multiclass", num_classes=3)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.val_acc(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        # 返回 dict 格式，让 Lightning 自动调用 scheduler 并配合 LearningRateMonitor 记录 lr
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ---------------------------------------------------------------
# 自定义 Callback：在训练结束时打印统计信息
# ---------------------------------------------------------------
class InfoPrinter(Callback):
    def on_train_start(self, trainer, pl_module):
        # pl_module ：当前的 LightningModule 模型
        print(">>> 自定义回调：训练开始")

    def on_train_epoch_end(self, trainer, pl_module):
        print(f">>> 自定义回调：epoch {trainer.current_epoch} 结束")

    def on_train_end(self, trainer, pl_module):
        print(">>> 自定义回调：训练结束，最好的 val_acc 日志见 checkpoint 或日志")


def make_data():
    x = torch.randn(1500, 8)
    w = torch.randn(8, 3)
    y = (x @ w).argmax(dim=1)
    ds = TensorDataset(x, y)
    train, val = torch.utils.data.random_split(ds, [1200, 300])
    return (
        DataLoader(train, batch_size=64, shuffle=True),
        DataLoader(val, batch_size=64),
    )


def main():
    train_loader, val_loader = make_data()

    # ModelCheckpoint：按 val_acc 保存最佳模型
    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-{epoch}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,  # 最多保存 1 个模型
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="mps",
        devices="auto",
        callbacks=[checkpoint, early_stop, lr_monitor, InfoPrinter()],
        log_every_n_steps=10,
    )

    trainer.fit(DemoModel(), train_loader, val_loader)
    print("最佳 checkpoint 路径:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()