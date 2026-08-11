"""
Lesson 4: LightningDataModule —— 把数据职责集中管理
=====================================================

学习目标：
  1. 掌握 LightningDataModule 的四件套：prepare_data / setup /
     train_dataloader / val_dataloader / test_dataloader。
  2. 理解 prepare_data 与 setup 的区别（下载/预处理 vs 切分/实例化）。
  3. 学会把 DataModule 传给 Trainer.fit()。

对应官方文档：
  - DataModules: https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
"""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchmetrics import Accuracy


# ---------------------------------------------------------------
# 1. 自定义一个简单的标准 Dataset（避免依赖外部下载）
# ---------------------------------------------------------------
class ToyDataset(Dataset):
    def __init__(self, n_samples: int, seed: int):
        self.x = torch.randn(n_samples, 8)
        w = torch.randn(8, 3, generator=torch.manual_seed(seed))
        self.y = (self.x @ w).argmax(dim=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------
# 2. LightningDataModule：把"数据"相关的所有逻辑集中到一个类里
# ---------------------------------------------------------------
class ToyDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        train_size: int = 2000,
        val_ratio: float = 0.2,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    # 只在单进程/单设备上执行一次：适合下载、tokenize、构建词表等
    def prepare_data(self):
        # 这里没有真正的下载，仅演示。可在此处做只执行一次的准备。
        # 只在 rank 0 跑一次：下载、tokenize、建词表……
        # 不要在这里 self.xxx = ... 赋值状态（其他 GPU 看不到）
        pass

    # 在每个进程上执行：负责真正切分数据、实例化 Dataset
    def setup(self, stage: str = None):
        # 每张卡都执行：切分数据、实例化 Dataset
        # 这里的 self.train_ds 等，在每张卡上各自存在一份
        if stage in (None, "fit"):
            full = ToyDataset(self.hparams.train_size, seed=0) # Dataset
            val_len = int(self.hparams.train_size * self.hparams.val_ratio)
            train_len = self.hparams.train_size - val_len
            self.train_ds, self.val_ds = random_split(full, [train_len, val_len])
        if stage in (None, "test"):
            self.test_ds = ToyDataset(200, seed=1)

    # 三个 dataloader 钩子
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


# ---------------------------------------------------------------
# 3. 一个配套的 LightningModule（复用 Trainer 的工程逻辑）
# ---------------------------------------------------------------
class Classifier(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))
        self.train_acc = Accuracy(task="multiclass", num_classes=3)
        self.val_acc = Accuracy(task="multiclass", num_classes=3)
        self.test_acc = Accuracy(task="multiclass", num_classes=3)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.train_acc(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.val_acc(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.test_acc(self(x), y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    dm = ToyDataModule(batch_size=64)
    model = Classifier()

    trainer = L.Trainer(max_epochs=5, accelerator="mps", devices="auto")
    # 只传 model 和 datamodule，Trainer 会自动调用它的 dataloaders
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()