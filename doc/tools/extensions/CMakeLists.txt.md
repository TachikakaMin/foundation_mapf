# `tools/extensions/CMakeLists.txt`

## 文件作用

这个文件定义了 `construct_features_native` Python 扩展的 CMake 构建方式。

## 主要配置

### 基础项目设置

- `project(MAPF_Extensions VERSION 1.0.0 LANGUAGES CXX)`
- `C++17`

### 依赖发现

- `Python3 Interpreter`
- `Python3 Development`
- `Python3 NumPy`

### 扩展模块

通过 `add_library(construct_features_native MODULE ...)` 构建扩展，并设置：

- 去掉前缀 `lib`
- 使用当前 Python 解释器对应的扩展后缀
- 开启隐藏符号可见性

### 编译与安装

- include 目录来自 Python 和 NumPy
- 链接 Python 运行库
- 编译参数包含：
  - `-O3`
  - `-ffast-math`
  - `-march=native`
- 安装路径是 `${Python3_SITELIB}`

### 自定义目标

- `build-extension`
- `test-extension`
- `clean-extension`
- `show-ext-help`

## 接口情况

### 产物

构建成功后会得到：

- `construct_features_native${PYTHON_EXT_SUFFIX}`

### 测试接口

`test-extension` 目标通过执行：

```bash
python -c "import construct_features_native; print('✅ 扩展导入成功')"
```

来验证扩展是否可导入。

## 用法

```bash
cd tools
mkdir -p build
cd build
cmake ..
make build-extension
make test-extension
```
