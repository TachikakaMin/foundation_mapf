# `MAPF_dataset.py`

## 文件作用

这个文件定义了面向单个 `.bin` 轨迹文件的数据集类 `MAPFDataset`。它把离线轨迹样本转换成训练/评估时可直接喂给模型的张量字典。

它主要被 [eval_test.py](/home/yimin/research/RAILGUN/eval_test.py) 使用，适合旧格式的单场景二进制数据。

## 主要接口

### `MAPFDataset(input_files, feature_dim, feature_type, first_step=False)`

- `input_files`: `.bin` 文件路径列表。
- `feature_dim`: 输入特征通道数。
- `feature_type`: 传给 `tools.utils.construct_input_feature` 的特征类型，常见值是 `gradient`。
- `first_step`: 为 `True` 时，每个文件只取第 0 步；否则把文件里的每个时间步都展开成一个样本。

## 类与方法

### `MAPFDataset.__init__(...)`

保存初始化参数，并立即调用 `load_file_info()` 预扫描数据文件。

### `MAPFDataset.load_file_info(first_step)`

扫描所有输入文件，构造两个索引数组：

- `file_indices`: 每个样本对应的文件名
- `step_indices`: 每个样本对应的时间步

`first_step=False` 时会并行读取文件头，用于得到每个 `.bin` 的步数。

### `MAPFDataset.load_single_file_info(file_name)`

读取单个 `.bin` 文件头部前 2 字节，返回该文件包含的总步数。这个方法只用于预扫描。

### `MAPFDataset.__getitem__(idx)`

按样本索引读取对应时间步的数据，并返回一个字典：

- `feature`: `torch.FloatTensor[feature_dim, H, W]`
- `action`: `torch.LongTensor[H, W]`
- `mask`: `torch.uint8[H, W]`，仅智能体所在位置为 1
- `file_name`: 原始 `.bin` 路径

处理流程是：

1. 从 `.bin` 中读取当前步的智能体位置和动作。
2. 从最后一步读取目标位置。
3. 用 `tools.utils.parse_file_name()` 推出对应地图。
4. 读取地图和距离图。
5. 调用 `tools.utils.construct_input_feature()` 生成模型输入。

### `MAPFDataset.__len__()`

返回样本总数，也就是展开后的时间步数量。

## 输入输出与接口情况

### 输入文件格式

文件按如下顺序读取：

- `uint16 steps`
- `uint16 agent_num`
- 对每个时间步：
  - `agent_num * 2` 字节的位置坐标
  - `agent_num` 字节的动作编码

### 依赖接口

- `tools.utils.read_map(map_path)`
- `tools.utils.read_distance_map(map_path)`
- `tools.utils.parse_file_name(file_name)`
- `tools.utils.construct_input_feature(...)`

## 用法

```python
from MAPF_dataset import MAPFDataset

dataset = MAPFDataset(
    input_files=["data/input_data/example.bin"],
    feature_dim=6,
    feature_type="gradient",
    first_step=True,
)
sample = dataset[0]
```
