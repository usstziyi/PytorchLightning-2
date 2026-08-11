"""
实战项目训练入口：CIFAR-10 完整训练流程
========================================
整合 Trainer、ModelCheckpoint、EarlyStopping、TensorBoardLogger、
DataModule 与模型，跑通一个完整的图像分类实战。

运行方式：
  uv run python 09_project/train.py
"""

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from cifar10_data import CIFAR10DataModule
from model import CIFAR10Model


def main():
    dm = CIFAR10DataModule(
        data_dir="./data",
        batch_size=128,
        num_workers=0,
        val_split=0.1,
    )

    model = CIFAR10Model(num_classes=10, lr=1e-3, t_max=10)

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="cifar10-{epoch}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    logger = TensorBoardLogger(save_dir="logs", name="cifar10")

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="mps",
        devices="auto",
        callbacks=[checkpoint, early_stop],
        logger=logger,
        log_every_n_steps=20,
    )

    # 训练 + 验证
    trainer.fit(model, datamodule=dm)

    # 在最佳 checkpoint 上测试
    if checkpoint.best_model_path:
        best_model = CIFAR10Model.load_from_checkpoint(checkpoint.best_model_path)
    else:
        best_model = model
    trainer.test(best_model, datamodule=dm)

    print(f"\n最佳模型已保存至: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()