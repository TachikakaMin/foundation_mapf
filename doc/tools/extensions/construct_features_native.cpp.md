# `tools/extensions/construct_features_native.cpp`

## 文件作用

这个文件实现了项目的原生 C++ 特征构造扩展，用于替代 Python 版 `construct_input_feature()` 的热点计算。

训练数据集 [MAPF_dataset_mbin.py](/home/yimin/research/RAILGUN/MAPF_dataset_mbin.py) 通过它来加速输入特征生成。

## 主要函数

### `get_distance_from_map(distance_map, agent_x, agent_y, goal_x, goal_y)`

统一距离查询入口，兼容两种 Python 侧对象：

- 带 `get_distance()` 方法的距离图对象
- Python 字典格式距离图

找不到时返回 `NOT_FOUND_PATH`。

### `construct_input_feature_cpp(self, args)`

暴露给 Python 的核心函数，实现了特征张量构造。

输入参数是：

- `map_data`
- `agent_locations`
- `goal_locations`
- `distance_map`
- `feature_dim`
- `feature_type`

返回：

- `numpy.float32[feature_dim, height, width]`

通道逻辑和 Python 版保持基本一致：

- `0`: 地图
- `1`: 智能体位置
- `2`: 目标位置
- `3`: 距离
- `4`: x 梯度
- `5`: y 梯度

其中梯度只在 `feature_type == "gradient"` 且 `feature_dim >= 5` 时计算。

### `PyInit_construct_features_native()`

模块初始化函数，负责初始化 NumPy C API 并返回 Python 模块对象。

## 模块接口

### 导出方法

通过 `module_methods` 对外只暴露一个 Python 方法：

- `construct_input_feature`

Python 包层再把它重命名为 `construct_input_feature_cpp`。

## 接口情况

### 被谁调用

- [tools/extensions/__init__.py](/home/yimin/research/RAILGUN/tools/extensions/__init__.py)
- [MAPF_dataset_mbin.py](/home/yimin/research/RAILGUN/MAPF_dataset_mbin.py)

### 输入约定

- `map_data` 是 `float32` 二维数组
- `agent_locations` / `goal_locations` 是 `int64` 二维数组
- `distance_map` 是 Python 对象，由 C API 回调查询

## 用法

正常情况下不直接手写导入底层模块，而是通过：

```python
from tools.extensions import construct_input_feature_cpp
```
