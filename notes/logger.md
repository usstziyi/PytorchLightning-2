`self.logger.experiment` 是 **TensorBoard 的 `SummaryWriter` 对象**（具体是 `torch.utils.tensorboard.SummaryWriter`）。

## 它是什么

`self.logger` 是 Lightning 的 `TensorBoardLogger`（一个封装壳），而 `.experiment` 属性暴露了它内部持有、真正干活的底层写入器。

```
self.logger            →  Lightning 的 TensorBoardLogger（封装 API）
self.logger.experiment →  底层 SummaryWriter（能直接调 add_scalar/add_image/add_histogram...）
```

## 验证一下类型

```python
print(type(self.logger))            # <class 'lightning.pytorch.loggers.tensorboard.TensorBoardLogger'>
print(type(self.logger.experiment)) # <class 'torch.utils.tensorboard.writer.SummaryWriter'>
```

`SummaryWriter` 正是 PyTorch 官方 `tensorboard` 库里的类，你可以直接 `from torch.utils.tensorboard import SummaryWriter` 创建的那种。

## 为什么需要 `.experiment`

Lightning 的高层 API（`self.log` / `self.log_dict`）只覆盖常用场景（标量）。像**图像、直方图、嵌入、PR 曲线**这类更丰富的可视化，Lightning 没封装，所以通过 `.experiment` 暴露底层 `SummaryWriter`，让你能直接调用它的 `add_*` 系列方法：

```python
self.logger.experiment.add_scalar(...)
self.logger.experiment.add_image(...)
self.logger.experiment.add_histogram(...)
self.logger.experiment.add_embedding(...)
```

## 两层 API 的关系

| 层 | 对象 | 用途 |
|---|---|---|
| Lightning 高层 | `self.log("train_loss", loss)` | 标量，自动聚合/同步 |
| 底层 | `self.logger.experiment.add_image(...)` | 图像/直方图等，自己控制 |

## 提醒

- 正因为是**底层 writer**，它不受 Lightning 的 `on_step`/`on_epoch` 逻辑管理，每次调用都直接写日志。
- 若换 logger（比如换 `WandbLogger`、`CSVLogger`），`.experiment` 指向的对象类型会变，`add_*` 方法签名可能不同——所以依赖 `.experiment` 的代码对 logger 类型有耦合。这也是为什么 Lightning 建议尽量用 `self.log`，只有需要底层能力时才用 `.experiment`。




三者的关系是：**`self.log` / `self.log_dict` 是高层封装，最终把数据交给 `self.logger` 去写进日志**。可以理解成"记账的人"和"账本"的关系。

## 一句话定位

```
self.log / self.log_dict  ──▶  self.logger  ──▶  磁盘上的日志文件
```

- `self.log` / `self.log_dict`：**决定记什么、怎么聚合**（高层 API）
- `self.logger`：**实际把数据写去哪**（底层存储）

## 逐个拆解

**1. `self.log(name, value, ...)`** — 记一个标量
```python
self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
```
- 最常用，记单个指标
- 优点：Lightning 自动处理**聚合、同步、检查点恢复**等

**2. `self.log_dict(dict, ...)`** — 一次记多个标量
```python
self.log_dict({"lr": ..., "grad_norm": ...}, on_step=True)
```
- 就是多个 `self.log` 的批量版，参数语义完全一样
- 内部会逐个调用 `self.log`

**3. `self.logger`** — 日志的"目的地"
```python
self.logger = TensorBoardLogger(save_dir="logs", name="lesson6")
```
- 是 `LightningModule` 的**属性**，值由 `Trainer(logger=...)` 注入
- 决定日志写到哪、用什么格式（TensorBoard / CSV / Wandb...）

## 完整数据流

```python
self.log("train_loss", loss)          # ① 高层：记录 train_loss
      │
      ▼  内部调用 self.logger.log_metrics(...)
self.logger (TensorBoardLogger)       # ② 写入器
      │
      ▼
logs/lesson6/events.out.tfevents.*    # ③ 落盘，TensorBoard 可读
```

火焰：`self.log` 调用了 `self.logger` 的 `log_metrics` 方法，把指标写进日志目录。

## 对比表

| | `self.log` | `self.log_dict` | `self.logger` |
|---|---|---|---|
| 是什么 | 方法 | 方法 | 对象/属性 |
| 记几个 | 1 个标量 | 多个标量 | — |
| 负责什么 | 聚合、同步、进度条 | 同左（批量） | 存储位置、格式 |
| 属于 | LightningModule 高层 API | 同左 | 由 Trainer 注入 |

## 关键点

- `self.log` 和 `self.log_dict` **不直接写磁盘**，它们负责"算好指标 + 交给 logger"。
- `self.logger` **不负责聚合**，只负责"把值写进文件"。
- 二者配合：`self.log` 计算并分发，`self.logger` 存储落地。你换 `self.log` 的参数（如 `on_epoch=True`）是改"怎么记录"，换 `Trainer(logger=CSVLogger(...))` 是改"记到哪"。

## 和 `.experiment` 的关系

上一题的 `self.logger.experiment` 是 `self.logger` 的**底层 writer**——当 `self.log` 覆盖不了（如图像、直方图）时，你绕过 `self.log`，直接通过 `self.logger.experiment.add_*()` 写底层 API。所以：

```
self.log  ──高层──▶  self.logger  ──底层──▶  self.logger.experiment
（记指标）          （写入器）               （TensorBoard 原始接口）
```