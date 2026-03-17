# `tools/utils.py`

## 文件作用

这个文件是项目的基础工具模块，负责：

- 距离查询兼容层
- 模型输入特征构造
- 数据文件名到地图路径的映射
- `.map` / 距离图读取
- 距离图预计算
- LACAM 坐标串解析

很多上层脚本都依赖它，包括数据集、rollout、可视化和距离图预计算。

## 常量

### `NOT_FOUND_PATH = 2048`

表示不可达或未找到的距离。

## 主要函数

### `get_distance(distance_map, agent_location, goal_location)`

统一的距离查询接口，兼容两种距离图表示：

- `DistanceMapReader` 这类带 `get_distance()` 方法的对象
- Python 字典格式 `{start_pos: dist_matrix}`

找不到时返回 `NOT_FOUND_PATH`。

### `construct_input_feature(map_data, agent_locations, goal_locations, distance_map, feature_dim, feature_type, previous_agent_locations=None)`

构造模型输入特征张量，是整个项目最关键的接口之一。

基础通道定义：

- `channel 0`: 地图障碍物
- `channel 1`: 智能体位置，值为 `agent_id + 1`
- `channel 2`: 目标位置，值为 `agent_id + 1`

额外通道按 `feature_dim` 变化：

- `feature_dim == 4`
  - `channel 3`: 当前点到目标的距离
- `feature_dim == 5` 且 `feature_type == "gradient"`
  - `channel 3`: x 方向梯度
  - `channel 4`: y 方向梯度
- `feature_dim == 6`
  - `channel 3`: 距离
  - `channel 4/5`: 梯度，或在非 `gradient` 模式下写入目标相对位移
- `feature_dim == 7`
  - `channel 6`: 上一步智能体位置

这个函数被 [MAPF_dataset.py](/home/yimin/research/RAILGUN/MAPF_dataset.py)、[tools/path_formation.py](/home/yimin/research/RAILGUN/tools/path_formation.py) 等直接调用。

### `parse_file_name(file_name)`

把轨迹文件路径解析成：

- `map_file_path`
- `path_name`

当前兼容三类命名体系：

- `mapf_gpt`
- `data_benchmark`
- 当前仓库使用的 `maze-...` 命名

### `read_map(map_file_path)`

读取 `.map` 文件并转成 `numpy` 二值矩阵：

- `1`: 障碍
- `0`: 可通行

### `read_distance_map(map_file_path)`

按优先级读取距离图：

1. `distance_maps/*.dmap`
2. `distance_maps/*.pkl`

返回值可能是：

- `DistanceMapReader`
- Python `pickle` 字典

### `calculate_single_point_distances(args)`

从一个起点出发做 BFS，得到到所有网格的最短距离矩阵。它是距离图预计算的底层 worker。

### `create_distance_map(map_data)`

为地图上每个可通行点调用 `calculate_single_point_distances()`，最终生成 `{start: dist_matrix}` 字典。

### `parse_coordinates(coord_str)`

解析 LACAM 结果中的 `(x,y)` 坐标串，并交换为项目内部统一使用的 `(row, col)`。

## 接口情况

### 典型依赖关系

- 数据集模块依赖 `read_map()`、`read_distance_map()`、`construct_input_feature()`
- 可视化模块依赖 `read_map()`、`parse_file_name()`
- 路径转换模块依赖 `parse_coordinates()`
- 预计算脚本依赖 `create_distance_map()`

## 用法

```python
from tools.utils import read_map, read_distance_map, construct_input_feature

map_data = read_map("data/map_files/example/example.map")
distance_map = read_distance_map("data/map_files/example/example.map")
feature = construct_input_feature(map_data, agent_locations, goal_locations, distance_map, 6, "gradient")
```
