# `models/unet_util.py`

## 文件作用

这个文件提供 `UNet` 使用的基础模块，分别对应编码、解码和输出头。它本身不直接作为训练入口，而是被 [models/unet.py](/home/yimin/research/RAILGUN/models/unet.py) 组合调用。

## 类与方法

### `DoubleConv(in_channels, out_channels, mid_channels=None)`

两层连续卷积块：

- `Conv2d -> BatchNorm2d -> ReLU`
- `Conv2d -> BatchNorm2d -> ReLU`

#### `DoubleConv.forward(x)`

输入输出的空间分辨率保持不变，通道数最终变为 `out_channels`。

### `Down(in_channels, out_channels)`

下采样块：

- `MaxPool2d(2)`
- `DoubleConv(in_channels, out_channels)`

#### `Down.forward(x)`

把特征图高宽减半，并提升通道数。

### `Up(in_channels, out_channels, bilinear=True)`

上采样块。

- `bilinear=True` 时使用 `Upsample`
- 否则使用 `ConvTranspose2d`

之后会把上采样结果与编码器特征做拼接，再交给 `DoubleConv`。

#### `Up.forward(x1, x2)`

- `x1`: 低分辨率特征
- `x2`: 编码器侧跳连特征

它会先上采样 `x1`，再通过 `F.pad` 修正尺寸差，最后沿通道维拼接。

### `OutConv(in_channels, out_channels)`

输出头，使用 `1x1 Conv` 做通道映射。

#### `OutConv.forward(x)`

把最后一层特征转成动作类别 logits。

## 接口情况

### 被谁调用

- [models/unet.py](/home/yimin/research/RAILGUN/models/unet.py)

### 形状约定

- 编码路径逐层减半空间尺寸
- 解码路径恢复到输入尺寸
- `OutConv` 不改变空间分辨率

## 用法

通常不单独直接使用，而是作为 `UNet` 的构件：

```python
from models.unet_util import DoubleConv, Down, Up, OutConv
```
