# `tools/distance_map_reader.py`

## 文件作用

这个文件实现了 `.dmap` 二进制距离图读取器，主要服务于 C++ 预处理后的距离图加载。

它的目标是替代大体积的 Python `pickle` 距离图，提高读取效率并统一距离查询接口。

## 类与函数

### `DistanceMapReader(dmap_file)`

读取 `.dmap` 文件并把内容加载到内存。

初始化后维护以下状态：

- `distance_map`
- `height`
- `width`

### `DistanceMapReader._load_distance_map()`

内部加载函数，按二进制格式依次读取：

- 有效起点数量
- 地图高宽
- 每个起点对应的整张距离矩阵

### `DistanceMapReader.get_distance(agent_location, goal_location)`

查询单个起点到目标点的距离；不存在或越界时返回 `NOT_FOUND_PATH`。

### `DistanceMapReader.get_all_distances_from(start_location)`

返回一个起点到全图的距离矩阵拷贝。

### `DistanceMapReader.is_valid_position(position)`

判断某个起点是否在距离图中存在。

### `DistanceMapReader.get_valid_positions()`

返回全部可查询的起点坐标。

### `DistanceMapReader.get_map_size()`

返回 `(height, width)`。

### `read_distance_map_cpp(map_file_path)`

把 `.map` 路径转换成 `.dmap` 路径，并返回 `DistanceMapReader` 实例。

### `get_distance_cpp(distance_map, agent_location, goal_location)`

对 `DistanceMapReader.get_distance()` 的轻量包装，便于兼容旧接口。

### `create_distance_map_dict_from_cpp(dmap_file)`

把 `.dmap` 内容重新暴露成 Python 字典格式，主要用于兼容性。

## 接口情况

### 对外接口

- [tools/utils.py](/home/yimin/research/RAILGUN/tools/utils.py) 中 `read_distance_map()`
- [tools/cached_distance_reader.py](/home/yimin/research/RAILGUN/tools/cached_distance_reader.py)
- C++ 特征构造扩展通过 Python 层对象方法调用 `get_distance()`

### `.dmap` 文件格式

按当前实现读取为：

- `uint32 num_positions`
- `uint32 height`
- `uint32 width`
- 对每个起点：
  - `uint16 start_x`
  - `uint16 start_y`
  - `height * width` 个 `uint16` 距离

## 用法

```python
from tools.distance_map_reader import DistanceMapReader

reader = DistanceMapReader("data/distance_maps/example/example.dmap")
distance = reader.get_distance((1, 1), (10, 10))
```
