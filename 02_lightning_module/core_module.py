"""
Lesson 2: LightningModule 核心
================================

学习目标：
  1. 掌握 LightningModule 中最常用的 hook（钩子）：
       forward / training_step / validation_step / test_step / predict_step
       configure_optimizers / configure_trainable_parameters
       on_train_epoch_end 等生命周期钩子。
  2. 理解 forward 与 training_step 的区别与联系。
  3. 学习 self.log 的参数（prog_bar / on_step / on_epoch / reduce_fx）。

对应官方文档：
  - LightningModule: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
"""

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CoreModule(L.LightningModule):
    """
    用一个简单的多分类问题演示核心钩子。
    强调：Lightning 会自动调用这些钩子，我们只需要"填空"。
    """

    def __init__(self, in_features: int = 8, n_classes: int = 3, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()  # 自动把 __init__ 的参数存到 self.hparams

        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    # ---------- 推理路径 ----------
    def forward(self, x):
        """推理时使用。返回原始 logits（不经过 softmax）。"""
        return self.net(x)

    # ---------- 训练 ----------
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 注意 acc 在 step 维度即可，flatten 到 epoch 维度可加 reduce_fx
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss  # 返回 loss 给 Trainer 用于反向传播

    # ---------- 验证 ----------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        # on_epoch=True 表示除了 step 级，还会聚合成 epoch 级指标
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, reduce_fx="mean")
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    # ---------- 测试 ----------
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        return acc

    # ---------- 预测 ----------
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """只做前向，返回预测类别。"""
        x, _ = batch
        logits = self(x)
        return logits.argmax(dim=1)

    # ---------- 优化器 ----------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ---------- 生命周期钩子示例 ----------
    def on_train_start(self):
        print("======== 训练开始 ========")

    def on_train_epoch_end(self):
        # 每个 epoch 结束时调用，可在这里做 epoch 级聚合逻辑
        print(f"  [epoch {self.current_epoch} 结束]")

    def on_train_end(self):
        print("======== 训练结束 ========")


def make_synthetic_data(num_samples: int = 2000, in_features: int = 8, n_classes: int = 3):
    """生成一个人造可分类数据集，避免依赖外部下载。"""
    torch.manual_seed(42)
    x = torch.randn(num_samples, in_features)  # (2000,8)
    # 用一个随机线性映射 + 三层中心，制造可分结构
    w = torch.randn(in_features, n_classes)  # (8,3)
    logits = x @ w  # (2000,3)
    y = logits.argmax(dim=1)  # (2000,)
    return TensorDataset(x, y)


def main():
    dataset = make_synthetic_data()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CoreModule()

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="mps",
        devices="auto",
        log_every_n_steps=10,
    )

    # fit 时同一个 loader 既当训练也当验证，仅用于演示
    trainer.fit(model, loader, val_dataloaders=loader)
    trainer.test(model, dataloaders=loader)

    # 预测演示
    preds = trainer.predict(model, dataloaders=loader)
    sample = preds[0][:10]
    print("预测类别示例:", sample.tolist())


if __name__ == "__main__":
    main()