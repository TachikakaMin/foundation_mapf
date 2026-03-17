# `tools/CMakeLists.txt`

## 文件作用

这是 `tools/` 的顶层 CMake 配置文件，用于组织扩展构建、全局构建目标、清理目标和打包元信息。

## 主要配置项

### 基础配置

- `cmake_minimum_required(VERSION 3.12)`
- `project(MAPF_Toolsuite VERSION 2.0.0 LANGUAGES CXX)`
- 使用 `C++17`
- 默认构建类型为 `Release`

### 选项

- `BUILD_EXTENSIONS`
- `BUILD_TOOLS`
- `BUILD_TESTS`

当前真正接入构建逻辑的是 `BUILD_EXTENSIONS`，它会决定是否 `add_subdirectory(extensions)`。

### 自定义目标

- `build-all`
- `clean-global`
- `show-help`

### 安装规则

文件中声明了两个安装脚本：

- `core/mapf_tools.py`
- `core/status_check.py`

它们会被安装到 `bin/`。这表示该 CMake 配置预留了更完整的工具集入口。

## 接口情况

### 与扩展子目录的关系

当 `BUILD_EXTENSIONS=ON` 时，会进入 [tools/extensions/CMakeLists.txt](/home/yimin/research/RAILGUN/tools/extensions/CMakeLists.txt) 构建 Python 扩展。

### 输出信息

配置阶段会打印：

- 构建类型
- 编译器
- `BUILD_TOOLS`
- `BUILD_EXTENSIONS`
- `BUILD_TESTS`

## 用法

```bash
cd tools
mkdir -p build
cd build
cmake ..
make build-all
```
