# `models/CNN.py`

## 文件作用

这个文件定义了一个纯卷积残差模型 `CNN`，用于把地图级输入特征映射为每个网格位置的动作分类结果。

它可作为 [train.py](/home/yimin/research/RAILGUN/train.py) 中 `--model cnn` 的实现。

## 类与函数

### `CNN(n_channels, n_classes)`

主模型类，结构如下：

1. `conv1 + bn1 + relu`
2. 8 个 `ResidualBlock`
3. `final_conv`
4. `softmax`

### `CNN.forward(x)`

- 输入：`x`，形状为 `[B, n_channels, H, W]`
- 输出：
  - `logits`: `[B, n_classes, H, W]`
  - `prob`: softmax 后的概率图

训练脚本默认用 `logits` 计算损失，`prob` 主要用于推理或调试。

### `CNN.save_model(path)`

把当前 `state_dict()` 保存到磁盘。

### `CNN.load_model(path)`

从权重文件恢复 `state_dict()`。

### `ResidualBlock(in_channels, out_channels)`

残差块实现：

- 主分支：两层 `3x3 Conv + BN`，中间带 `ReLU`
- 捷径分支：当输入输出通道不一致时使用 `1x1 Conv + BN`

### `ResidualBlock.forward(x)`

执行残差加和并做一次 `ReLU`。

### `count_parameters(model)`

统计可训练参数总数。

## 接口情况

### 对外接口

- [train.py](/home/yimin/research/RAILGUN/train.py) 通过 `CNN(...).to(device)` 使用它
- 模型的前向返回格式和 `UNet` 保持一致，便于训练脚本复用

### 输入输出约定

- `n_channels` 一般来自 `args.feature_dim`
- `n_classes` 一般来自 `args.action_dim`

## 用法

```python
from models.CNN import CNN

model = CNN(n_channels=6, n_classes=5)
logits, prob = model(x)
```
