# `models/unet.py`

## 文件作用

这个文件定义了项目默认使用的 `UNet` 模型。它把地图特征编码为多尺度表示，再通过跳连解码为每个网格的动作分布。

训练入口 [train.py](/home/yimin/research/RAILGUN/train.py) 默认选择这个模型。

## 类与方法

### `UNet(n_channels, n_classes, first_layer_channels=64, bilinear=False, blocks_per_stage=1)`

构造完整 U-Net：

- 编码器：
  - `input_conv`
  - `down1`
  - `down2`
  - `down3`
  - `down4`
- 解码器：
  - `up1`
  - `up2`
  - `up3`
  - `up4`
- 输出层：
  - `output_conv`
  - `softmax`

参数含义：

- `n_channels`: 输入通道数
- `n_classes`: 动作类别数
- `first_layer_channels`: 第一层基础通道数，是主要模型规模控制旋钮之一
- `bilinear`: 是否用双线性上采样替代反卷积
- `blocks_per_stage`: 每个 stage 的 `ResBlock` 数；`0` 时退回旧版 `DoubleConv`

当前结构约定：

- `blocks_per_stage > 0` 时，`input_conv` / `Down` / `Up` 都会使用 `ResStage`
- `blocks_per_stage == 0` 时，模型退回兼容旧版的 `DoubleConv` 风格 U-Net

因此现在的 UNet 同时支持：

- 改宽：调 `first_layer_channels`
- 改深：调 `blocks_per_stage`

这两个参数也是 `scaling_law.py` 里默认扫描的模型大小轴。

### `UNet.forward(x)`

- 输入：`[B, n_channels, H, W]`
- 输出：
  - `logits`: `[B, n_classes, H, W]`
  - `prob`: softmax 概率图

### `UNet.use_checkpointing()`

意图是把各子模块替换为 checkpoint 形式，以减少显存占用。这个接口没有在训练主流程中被调用。

## 接口情况

### 依赖模块

所有子模块都来自 [models/unet_util.py](/home/yimin/research/RAILGUN/models/unet_util.py)：

- `ResBlock`
- `ResStage`
- `DoubleConv`
- `Down`
- `Up`
- `OutConv`

### 输入尺寸

文件注释写明最小可处理尺寸为 `16 x 16`。训练脚本当前又额外过滤掉了小于 `32 x 32` 的地图。

## 用法

```python
from models.unet import UNet

model = UNet(
    n_channels=6,
    n_classes=5,
    first_layer_channels=64,
    bilinear=False,
    blocks_per_stage=1,
)
logits, prob = model(x)
```
