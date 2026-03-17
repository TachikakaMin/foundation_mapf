# `tools/extensions/__init__.py`

## 文件作用

这个文件把编译后的原生扩展重新导出为 Python 层接口。

## 主要接口

### `construct_input_feature_cpp`

通过下面这句导入：

```python
from .construct_features_native import construct_input_feature as construct_input_feature_cpp
```

也就是说，外部模块看到的是 Python 风格的函数名 `construct_input_feature_cpp`，底层真正实现来自编译生成的 `construct_features_native` 模块。

## 接口情况

- `__all__ = ['construct_input_feature_cpp']`
- 如果扩展还没有编译，这个文件在导入时就会报错

## 用法

```python
from tools.extensions import construct_input_feature_cpp
```
