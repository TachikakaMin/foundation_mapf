# `tools/build.sh`

## 文件作用

这是 `tools/` 目录下的统一构建脚本，用来：

- 编译 C++ 路径转换工具
- 配置并构建 CMake 扩展
- 清理构建目录
- 运行测试
- 安装构建产物

## 主要函数

### `print_header()`

打印脚本标题和版本信息。

### `print_usage()`

打印支持的子命令和常用 CMake 选项。

### `build_path_converter()`

单独编译 `convert_path_to_mbin.cpp`，生成：

- `tools/convert_path_to_mbin`

流程：

1. 检查 `g++`
2. 在 `tools/` 目录执行编译命令
3. 输出可执行文件路径和使用提示

### `build_project()`

完整构建流程：

1. 先尝试编译路径转换工具
2. 创建 `tools/build/`
3. 执行 `cmake -DPython3_EXECUTABLE=$(which python) ..`
4. 执行 `make -j$(nproc)`
5. 如果扩展 `.so` 存在，则复制回 `tools/extensions/`

### `clean_project()`

清理：

- `tools/build/`
- `tools/convert_path_to_mbin`

### `run_tests()`

进入构建目录并执行 `make test`。

### `install_project()`

进入构建目录并执行 `sudo make install`。

## 脚本入口

默认子命令是 `build`，支持：

- `build`
- `clean`
- `rebuild`
- `test`
- `install`
- `converter`
- `help`

## 接口情况

### 依赖

- `bash`
- `g++`
- `cmake`
- `make`
- `nproc`
- 当前激活的 Python 环境

### 产物

- `tools/convert_path_to_mbin`
- `tools/build/`
- `tools/extensions/construct_features_native*.so`

## 用法

```bash
cd tools
bash build.sh build
bash build.sh converter
bash build.sh clean
```
