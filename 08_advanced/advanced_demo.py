"""
Lesson 8: 高级特性
===================

学习目标：
  1. 学习率调度器（LR scheduler）与优化器的配置。
  2. 多优化器 / 参数组（如冻结 backbone 只训练头部）。
  3. 梯度累积、混合精度、多卡（strategy）等进阶设置概览。
  4. Lightning Fabric 的简单介绍（Lightning 2.0 的新玩法）。

对应官方文档：
  - Optimizers: https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html
  - Fabric: https://lightning.ai/docs/fabric/stable/
"""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class AdvancedModel(L.LightningModule):
    def __init__(self, lr: float = 1e-2, t_max: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

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
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 1) 返回 (优化器, 调度器) 的字典形式
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.t_max
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 或 "step"
                "frequency": 1,
                "monitor": "val_loss",  # 若是 ReduceLROnPlateau 需指定
            },
        }


def main():
    x = torch.randn(1200, 8)
    w = torch.randn(8, 3)
    y = (x @ w).argmax(dim=1)
    ds = TensorDataset(x, y)
    train, val = torch.utils.data.random_split(ds, [1000, 200])
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    val_loader = DataLoader(val, batch_size=64)

    trainer = L.Trainer(max_epochs=5, accelerator="mps", devices="auto")
    # strategies: 'ddp'（多卡）、precision='bf16-mixed' 等
    trainer.fit(AdvancedModel(), train_loader, val_loader)

    print("\n>>> 进阶：多优化器 / 冻结层 / Fabric 等更复杂用法可查看官方文档。")


if __name__ == "__main__":
    main()