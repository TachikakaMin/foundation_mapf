# `tools/extensions/__init__.py`

## 文件作用

这个文件把编译后的原生扩展重新导出为 Python 层接口。

## 主要接口

### `construct_input_feature_cpp`

通过下面这句导入：

```python
from .construct_features_native import construct_input_feature as construct_input_feature_cpp
```

外部模块看到的是 Python 风格的函数名 `construct_input_feature_cpp`，底层真正实现来自编译生成的 `construct_features_native` 模块。

### `generate_lacam_solution_cpp(map_file, agent_num, seed, time_limit_sec=2, verbose=0)`

这个接口由：

```python
from .lacam_online_native import generate_lacam_solution_cpp
```

导出，用于在线训练时直接在内存里调用 LACAM，返回一整条场景轨迹。

如果 `lacam_online_native` 还没有编译完成，这个文件会提供一个兜底函数，在调用时抛出 `ImportError` 并提示先运行构建脚本。

## 接口情况

- `__all__ = ["construct_input_feature_cpp", "generate_lacam_solution_cpp"]`
- `construct_features_native` 是硬依赖
- `lacam_online_native` 采用延迟失败策略：包可导入，但一旦调用未构建的在线生成函数会报错

## 用法

```python
from tools.extensions import construct_input_feature_cpp, generate_lacam_solution_cpp

scenario = generate_lacam_solution_cpp(
    "data/map_files/maze-32-32-10-1-75/maze-32-32-10-1-75-0.map",
    16,
    123,
    time_limit_sec=2,
)
```
