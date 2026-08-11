# 关于 torchmetrics 的 `compute()`

`compute()` 是 torchmetrics 指标对象的**核心方法之一**，用于把内部累积的原始统计量换算成最终的指标值。要真正理解它，得先知道指标的完整生命周期。

## 一、指标的三个核心方法

每个 `Metric` 都有三个方法：

| 方法 | 作用 | 是否改变状态 |
|------|------|-------------|
| `update()` | 用一个新的 batch 更新内部**累积统计量** | 是（累积） |
| `compute()` | 根据已累积的数据**算出最终指标值** | 否（只读） |
| `reset()` | 清空累积的统计量 | 是（归零） |

对应到你的代码：

- `self.train_acc(logits, y)` → 等价于 `self.train_acc.update(logits, y)`（`__call__` 内部会调用 update）
- `self.train_acc.compute()` → 算出当前累积的准确率
- `self.train_acc.reset()` → 清空，准备下一轮

## 二、`Accuracy` 内部到底累积了什么

多分类 `Accuracy` 内部会累积两个东西（都是张量，可跨 batch 累加）：

- `correct`：预测正确的样本总数
- `total`：参与统计的样本总数

`compute()` 的逻辑大致是：

```python
def compute(self):
    return self.correct / self.total   # 简单平均，不受 batch 大小影响
```

这就是关键区别：它**不是**对每个 batch 的准确率做算术平均，而是 `总正确数 / 总样本数`。因此即使某个 batch 大小不同，最终结果也是准确的。

## 三、在你的代码里，compute 是谁调用的

**你不需要手动调用 `compute()`**，是 Lightning 帮你调用的。看这一行：

```python
self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
```

这里传给 `self.log` 的是一个**指标对象**（不是数字）。Lightning 检测到参数是 `Metric` 实例后，会在合适时机自动：

1. 调用 `self.train_acc.compute()` 拿到当前准确率；
2. 把数值记录到日志（TensorBoard / 进度条）；
3. 在合适的边界调用 `self.train_acc.reset()`。

## 四、compute 什么时候被调用

这取决于 `self.log` 的 `on_step` / `on_epoch` 参数：

| 场景 | compute 时机 |
|------|-------------|
| `on_epoch=True`（你这里的用法） | 每个 **epoch 结束**时 compute 一次，得到整个 epoch 的准确率 |
| `on_step=True` | 每个 **batch** 结束时 compute 一次，得到截止当前的累积准确率 |
| `prog_bar=True` | compute 出的值会显示到 tqdm 进度条上 |

以你的 `train_acc` 为例（`on_step=True, on_epoch=True`）：

- **每个 batch**：`update` 累积 → `compute` 得到"截止目前"的准确率 → 画进度条/记 step 日志
- **每个 epoch 结束**：`compute` 得到整个 epoch 的最终准确率 → 记录 epoch 日志 → **`reset()` 清空**，为下一 epoch 重新累积

## 五、关键点：reset 也很重要

`compute` 是**只读**的，它不会清空累积数据。如果不在 epoch 边界 reset，准确率会一直累加下去（把前面所有 epoch 的数据混在一起）。

Lightning 在 `on_epoch=True` 的日志记录流程里，compute 完之后会自动 `reset()`。这跟你上一轮问的"为什么要三个独立对象"相辅相成：

- **三个对象** → 训练/验证/测试各管各的累积，互不污染；
- **compute + reset** → 同一对象在**不同 epoch** 之间也能正确隔离。

## 六、什么时候需要手动调用 compute

只有当你**不用 `self.log` 传指标对象**、而是想自己拿数值时才需要：

```python
# 手动方式（不推荐，Lightning 会管）
self.train_acc(logits, y)          # update
acc = self.train_acc.compute()     # 手动算
self.train_acc.reset()             # 手动清
```

或者调试时想打印当前准确率：

```python
print(self.val_acc.compute())
```

---

**一句话总结**：`compute()` 是把指标内部累积的原始统计量（`correct/total`）换算成最终准确率的**只读方法**；在 Lightning 里你只需 `update`（即 `self.train_acc(...)`），compute 和 reset 都由 `self.log` 自动管理，你不需要（也不应该）手动调用。