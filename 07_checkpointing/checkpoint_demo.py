"""
Lesson 7: 模型保存与加载
=========================

学习目标：
  1. 理解 Lightning 的 checkpoint 机制：保存权重 + 超参数 + 优化器状态 + epoch 等。
  2. 学会用 ModelCheckpoint 自动保存，以及手动的 trainer.save_checkpoint。
  3. 学会如何从 checkpoint 恢复训练（resume）和加载推理（load_from_checkpoint）。

对应官方文档：
  - Checkpointing: https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html
"""

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class SaveModel(L.LightningModule):
    def __init__(self, hidden_size: int = 32, lr: float = 1e-2):
        super().__init__()
        self.save_hyperparameters()  # 保存 hidden_size 和 lr
        self.net = nn.Sequential(nn.Linear(8, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 3))

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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def make_data():
    x = torch.randn(1200, 8)
    w = torch.randn(8, 3)
    y = (x @ w).argmax(dim=1)
    ds = TensorDataset(x, y)
    train, val = torch.utils.data.random_split(ds, [1000, 200])
    return DataLoader(train, batch_size=64, shuffle=True), DataLoader(val, batch_size=64)


def main():
    train_loader, val_loader = make_data()

    ckpt = ModelCheckpoint(
        dirpath="checkpoints",
        filename="save-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="mps",
        devices="auto",
        callbacks=[ckpt],
    )
    model = SaveModel(hidden_size=32)
    trainer.fit(model, train_loader, val_loader)

    best_path = ckpt.best_model_path
    print("\n最佳 checkpoint:", best_path)

    # ---------- 方式 1：从 checkpoint 恢复为可推理的模型 ----------
    # 因为 save_hyperparameters 保存了超参，load_from_checkpoint 会自动重建模型
    restored = SaveModel.load_from_checkpoint(best_path)
    restored.eval()
    with torch.no_grad():
        # sample 必须和模型在同一设备上，否则报 "input is on cpu but expected on mps"
        sample = torch.randn(5, 8, device=restored.device)
        preds = restored(sample).argmax(dim=1)
    print("恢复模型推理结果:", preds.tolist())

    # ---------- 方式 2：从 checkpoint 恢复训练（resume） ----------
    trainer2 = L.Trainer(
        max_epochs=5,  # 注意：resume 会从 checkpoint 的 epoch 继续，此处仅演示
        accelerator="mps",
        devices="auto",
    )
    trainer2.fit(SaveModel(hidden_size=32), train_loader, val_loader, ckpt_path=best_path)


if __name__ == "__main__":
    main()