# `MAPF_dataset_mbin.py`

## 文件作用

这个文件实现了面向 `.mbin` 合并数据的 `MAPFDataset`。相比 [MAPF_dataset.py](/home/yimin/research/RAILGUN/MAPF_dataset.py)，它支持一个文件内包含多个场景，并且默认走缓存距离图和 C++ 特征构建扩展，适合训练阶段的大规模数据加载。

它主要被 [train.py](/home/yimin/research/RAILGUN/train.py) 使用。

## 主要接口

### `MAPFDataset(input_files, feature_dim, feature_type, first_step=False)`

- `input_files`: `.mbin` 文件路径列表。
- `feature_dim`: 输入特征通道数。
- `feature_type`: 特征构建模式，通常是 `gradient`。
- `first_step`: 为 `True` 时，每个文件只取 `scenario 0 / step 0`；否则展开所有场景的所有时间步。

## 类与方法

### `MAPFDataset.__init__(...)`

初始化索引数组，并调用 `load_file_info()` 扫描 `.mbin` 文件。

### `MAPFDataset.load_file_info(first_step)`

生成三类样本索引：

- `file_indices`: 对应 `.mbin` 文件
- `step_indices`: 对应场景内时间步
- `scenario_indices`: 对应场景编号

### `MAPFDataset.load_single_merged_file_info(file_name)`

读取 `.mbin` 头部和索引表，返回该文件中每个场景的步数信息，用于后续展开样本。

### `MAPFDataset.parse_map_name_from_mbin(file_name)`

根据 `.mbin` 文件名或目录结构反推出地图文件路径。优先按命名规则拼接 `data/map_files/.../*.map`，找不到时再做目录级回退。

### `MAPFDataset.__getitem__(idx)`

读取一个具体的 `scenario + step`，返回值与旧版数据集保持一致：

- `feature`
- `action`
- `mask`
- `file_name`

和旧版的差别在于：

- 地图距离图通过 `tools.cached_distance_reader.read_distance_map_cached()` 缓存
- 特征通过 `tools.extensions.construct_input_feature_cpp()` 在 C++ 中构造

### `MAPFDataset.__len__()`

返回展开后的样本总数。

## 输入输出与接口情况

### `.mbin` 文件结构

脚本按下面的格式读取：

- 16 字节文件头，其中前 4 字节是 `num_scenarios`
- `num_scenarios * 272` 字节索引表，每条记录包含：
  - `offset(8)`
  - `data_size(4)`
  - `steps(2)`
  - `agent_num(2)`
  - `file_name(256)`
- 每个场景数据块内部再按 `.bin` 的单场景格式存放

### 依赖接口

- `tools.utils.read_map(map_path)`
- `tools.cached_distance_reader.read_distance_map_cached(map_path)`
- `tools.extensions.construct_input_feature_cpp(...)`

## 用法

```python
from MAPF_dataset_mbin import MAPFDataset

dataset = MAPFDataset(
    input_files=["data/input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-16/maze-32-32-10-1-75-16.mbin"],
    feature_dim=6,
    feature_type="gradient",
)
sample = dataset[0]
```
