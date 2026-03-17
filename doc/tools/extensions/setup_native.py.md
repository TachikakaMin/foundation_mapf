# `tools/extensions/setup_native.py`

## 文件作用

这个文件提供基于 `distutils` 的原生扩展编译配置，是不用 CMake 时的另一种构建入口。

## 主要配置

### `ext_modules`

定义了一个 `Extension`：

- 模块名：`construct_features_native`
- 源文件：`construct_features_native.cpp`
- 头文件目录：`numpy.get_include()`
- 语言：`c++`
- 编译参数：
  - `-O3`
  - `-std=c++17`

### `setup(...)`

注册扩展模块并关闭 `zip_safe`。

## 接口情况

- 构建产物是可被 Python 导入的 `construct_features_native` 扩展模块
- 对应的 Python 重导出入口见 [tools/extensions/__init__.py](/home/yimin/research/RAILGUN/tools/extensions/__init__.py)

## 用法

```bash
cd tools/extensions
python setup_native.py build_ext --inplace
```
