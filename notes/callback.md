toolName: view_files
            
status: success
          
            
filePath: /Users/usst_ziyi/Programs/trae/DeepL/PytorchLightning-2/02_lightning_module/core_module.py
          
两处名字相同，但**一个在 Callback 里、一个在 LightningModule（模型）里**，是两套独立且并行的机制。区别如下：

## 1. 所属类不同

| | [callbacks_demo.py](file:///Users/usst_ziyi/Programs/trae/DeepL/PytorchLightning-2/05_callbacks/callbacks_demo.py#L61-70) | [core_module.py](file:///Users/usst_ziyi/Programs/trae/DeepL/PytorchLightning-2/02_lightning_module/core_module.py#L90-98) |
|---|---|---|
| 所在类 | `InfoPrinter(Callback)` | `DemoModel(L.LightningModule)` |
| 本质 | 工程逻辑（可插拔） | 模型/训练逻辑 |
| 方法签名 | `(self, trainer, pl_module)` | `(self)` |
| 谁来调用 | Trainer 调 **callback** | Trainer 调 **model hook** |

## 2. 签名差异 = 数据来源差异

- **Callback 版**：`InfoPrinter` 是**独立于模型的旁观者**，它不认识模型、也不认识 Trainer 状态，所以 Trainer 把 `trainer` 和 `pl_module` 两个参数**塞给它**。它才能访问 `trainer.current_epoch`。
- **Module 版**：`DemoModel` **自己就是模型**，`self` 既代表模型又代表状态，所以不需要额外传参，直接 `self.current_epoch`。

## 3. 触发时机相同，机制不同

两者都会在**同一个节点被 Trainer 触发**（训练开始 / epoch 结束 / 训练结束），但：

```
trainer.fit(...)
  ├── on_train_start:
  │     ├─ 调 model.on_train_start()          (core_module 版)
  │     └─ 调 callback.on_train_start(trainer, model)
  ├── on_train_epoch_end:
  │     ├─ 调 model.on_train_epoch_end()
  │     └─ 调 callback.on_train_epoch_end(trainer, model)
  └── ...
```

它俩是**同时都会被调用的两条独立通道**，互不覆盖、互不依赖。

## 4. 实际运行效果

如果两个文件都跑，你会看到**每个节点打印两行**（一行来自 model hook，一行来自 callback）。比如训练开始会输出：

```
======== 训练开始 ========      ← core_module 的 model hook
>>> 自定义回调：训练开始        ← callbacks_demo 的 callback
```

## 5. 该用哪个？

| 场景 | 用哪个 |
|------|--------|
| 修改模型内部（如记录某层激活、改 forward 行为）| 放 Module 钩子 |
| 与模型无关的通用能力（存 checkpoint、早停、打日志、进度条）| 放 Callback |
| 想让多个不同模型复用同一段逻辑 | Callback |

一句话总结：**名字相同，但一个是"模型自己的钩子"，一个是"外挂的通用回调"。Callback 因为要"外挂"，所以需要把 `trainer` 和 `pl_module` 显式传进来；Module 的钩子因为自己就是主角，`self` 就够了。**