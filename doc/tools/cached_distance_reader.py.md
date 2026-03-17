# `tools/cached_distance_reader.py`

## 文件作用

这个文件在 [tools/distance_map_reader.py](/home/yimin/research/RAILGUN/tools/distance_map_reader.py) 的基础上增加了全局缓存，避免训练时反复加载同一张地图的距离图。

它目前主要被 [MAPF_dataset_mbin.py](/home/yimin/research/RAILGUN/MAPF_dataset_mbin.py) 使用。

## 主要接口

### `CachedDistanceMapReader.get_distance_map(map_file_path)`

静态方法。读取顺序如下：

1. 先尝试 `.dmap`
2. 失败时回退到 `.pkl`
3. 成功加载后写入全局 `_distance_map_cache`

返回值可能是：

- `DistanceMapReader`
- Python `pickle` 字典

### `read_distance_map_cached(map_file_path)`

对 `CachedDistanceMapReader.get_distance_map()` 的便捷封装。

### `clear_distance_map_cache()`

清空全局缓存字典。

### `get_cache_info()`

返回缓存状态：

- `cached_maps`
- `cache_keys`

## 接口情况

### 缓存键

缓存键直接使用实际文件路径：

- 优先是 `.dmap` 路径
- 回退时是 `.pkl` 路径

### 依赖

- [tools/distance_map_reader.py](/home/yimin/research/RAILGUN/tools/distance_map_reader.py)
- `pickle`

## 用法

```python
from tools.cached_distance_reader import read_distance_map_cached, get_cache_info

distance_map = read_distance_map_cached("data/map_files/example/example.map")
print(get_cache_info())
```
