# PyTorch Lightning 循序渐进学习计划

> 基于 [官方文档](https://pytorch-lightning.readthedocs.io/en/stable/) 与
> [GitHub 仓库](https://github.com/Lightning-AI/pytorch-lightning) 整理。
> 每个知识点独立成文件，由浅入深，最后通过一个完整的 CIFAR-10 图像分类实战项目收尾。

## 环境与运行

本项目使用 `uv` 管理依赖，**无需手动安装环境**，写好 `pyproject.toml`，直接运行即可：

```bash
# 首次运行会自动创建虚拟环境并安装依赖
uv sync

# 运行某个 lesson
uv run python 01_hello_lightning/hello_lightning.py
```

依赖：`torch`、`torchvision`、`lightning`（PyTorch Lightning 2.x，导入名 `lightning`）、`tensorboard`。

> 本仓库所有 Trainer 均已显式配置 `accelerator="mps"`，可直接在 Apple Silicon
> （M 系列芯片）上使用 MPS 设备加速训练。Lesson 3 演示了按设备自动选择精度：
> CUDA 用 fp16、MPS 用 bf16、其它回退到 32。

## 学习路线总览

| 阶段 | 目录 | 主题 | 对应官方文档 |
|------|------|------|--------------|
| 入门 | `01_hello_lightning` | 最小可运行示例、核心概念 | [15 分钟入门](https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html) |
| 基础 | `02_lightning_module` | LightningModule 核心钩子 | [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) |
| 基础 | `03_trainer` | Trainer 详解 | [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) |
| 基础 | `04_datamodule` | LightningDataModule | [DataModules](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html) |
| 进阶 | `05_callbacks` | Callbacks 回调 | [Callbacks](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) |
| 进阶 | `06_logging` | 日志记录与可视化 | [Logging](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) |
| 进阶 | `07_checkpointing` | 模型保存与加载 | [Checkpointing](https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html) |
| 进阶 | `08_advanced` | 学习率调度、多优化器等高级特性 | [Optimization](https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html) |
| 实战 | `09_project` | CIFAR-10 图像分类完整项目 | 综合应用 |

## 各 Lesson 说明

### Lesson 1 — 初识 Lightning（15 分钟入门）
`hello_lightning.py`：7 行核心代码跑通 MNIST 分类。理解两个核心类：
- **LightningModule**：打包"研究代码"（模型 + 训练/验证/测试 step + 优化器）。
- **Trainer**：负责"工程代码"（循环、设备、进度条、日志）。

### Lesson 2 — LightningModule 核心
`core_module.py`：逐个演示核心钩子 `forward / training_step / validation_step /
test_step / predict_step / configure_optimizers` 及生命周期钩子，学习 `self.log` 的参数。

### Lesson 3 — Trainer 详解
`trainer_demo.py`：`accelerator / devices / max_epochs / precision /
gradient_clip_val / accumulate_grad_batches / overfit_batches` 等常用参数。

### Lesson 4 — LightningDataModule
`toy_datamodule.py`：把数据职责集中到 `prepare_data / setup` 与三个 dataloader 钩子，
并学会传入 `Trainer.fit(datamodule=...)`。

### Lesson 5 — Callbacks
`callbacks_demo.py`：`ModelCheckpoint`、`EarlyStopping`、`LearningRateMonitor` 等内置回调，
以及如何自定义一个 `Callback`。

### Lesson 6 — 日志与可视化
`logging_demo.py`：`self.log / self.log_dict`、`TensorBoardLogger`、记录直方图与图像。

### Lesson 7 — 模型保存与加载
`checkpoint_demo.py`：checkpoint 机制、`load_from_checkpoint` 恢复推理、`ckpt_path` 恢复训练。

### Lesson 8 — 高级特性
`advanced_demo.py`：学习率调度器、优化器字典写法、多卡/混合精度策略概览。

### 实战项目 — CIFAR-10 图像分类
`09_project/` 完整工程，综合运用以上所有知识点：
- `cifar10_data.py`：CIFAR-10 DataModule（数据增强、切分）。
- `model.py`：ResNet 风格残差 CNN + torchmetrics 指标 + 学习率调度。
- `train.py`：训练 + 验证 + 测试一体的入口。
- `predict.py`：加载 checkpoint 对新图片分类。

```bash
uv run python 09_project/train.py      # 训练（首次会自动下载 CIFAR-10）
uv run python 09_project/predict.py --ckpt checkpoints/cifar10-xxx.ckpt --image xxx.png
```

## 进阶方向
- 多 GPU 训练：`Trainer(strategy="ddp")`。
- [Lightning Fabric](https://lightning.ai/docs/fabric/stable/)：更轻量、可细粒度控制的训练框架。
- 官方示例库：[lightning-tutorials](https://github.com/Lightning-AI/lightning-tutorials)。
- 更多通用工作流：[Common Workflows](https://pytorch-lightning.readthedocs.io/en/stable/common_usecases.html)。