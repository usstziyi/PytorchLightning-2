"""
Lesson 6: 日志记录与可视化
===========================

学习目标：
  1. 认识 self.log / self.log_dict 的常用参数。
  2. 使用 TensorBoardLogger 记录标量、直方图、图像。
  3. 学会自定义 logger 或在 TrainingModule 中记录任意张量。

对应官方文档：
  - Loggers: https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
  - 可视化: https://pytorch-lightning.readthedocs.io/en/stable/visualization.html
"""

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class LoggingModel(L.LightningModule):
    def __init__(self, lr=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 记录标量
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # grad 在 backward 之后才存在，首个 step 需判空，避免 torch.stack([]) 报错
        grads = [p.grad.flatten() for p in self.parameters() if p.grad is not None]
        # 把列表里所有一维梯度 首尾相接，计算L2范数
        grad_norm = torch.norm(torch.cat(grads)) if grads else torch.tensor(0.0)

        # log_dict 批量记录
        self.log_dict(
            {
                "lr": self.optimizers().param_groups[0]["lr"],
                "grad_norm": grad_norm,
            },
            on_step=True,
        )

        # 记录直方图（权重分布）
        self.logger.experiment.add_histogram(
            "fc1.weight", # 标签名：直方图名称
            self.net[0].weight, # 要记录的值：权重张量
            self.global_step # 横轴：当前训练步
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        # 记录一张示例图像（仅演示，用随机数据代替真实图像）
        # self.logger            →  Lightning 的 TensorBoardLogger（封装 API）
        # self.logger.experiment →  底层 SummaryWriter（能直接调 add_scalar/add_image/add_histogram...）
        if hasattr(self.logger, "experiment"):
            fake_img = torch.rand(1, 3, 28, 28)
            self.logger.experiment.add_image(
                "sample_image", # 标签名：图像名称
                fake_img[0], # 要记录的值：随机图像
                self.current_epoch # 横轴：当前验证 epoch
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def make_data():
    x = torch.randn(1000, 8)
    w = torch.randn(8, 3)
    y = (x @ w).argmax(dim=1)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=64, shuffle=True)


def main():
    loader = make_data()

    # 决定日志写到哪、用什么格式（TensorBoard / CSV / Wandb...）
    logger = TensorBoardLogger(save_dir="logs", name="lesson6")

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="mps",
        devices="auto",
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(LoggingModel(), loader, val_dataloaders=loader)
    print("日志已保存，可运行: tensorboard --logdir logs")


if __name__ == "__main__":
    main()