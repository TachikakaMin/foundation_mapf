# `tools/extensions/CMakeLists.txt`

## 文件作用

这个文件定义了 `tools/extensions/` 下原生 Python 扩展的 CMake 构建方式。当前不只构建特征扩展，还会把 vendored `lacam3` 链接成在线数据生成扩展。

## 主要配置

### 基础项目设置

- `project(MAPF_Extensions VERSION 1.0.0 LANGUAGES CXX)`
- `C++17`

### 依赖发现

- `Python3 Interpreter`
- `Python3 Development`
- `Python3 NumPy`

### 扩展模块

当前包含两个模块：

- `construct_features_native`
- `lacam_online_native`

其中：

- `construct_features_native` 负责高性能输入特征构建
- `lacam_online_native` 负责在线调用 LACAM 生成完整轨迹

`lacam_online_native` 通过 `add_subdirectory(.../data_generation_LACAM/lacam3/lacam3)` 直接链接 vendored `lacam3` 静态库。

两个扩展都会设置：

- 去掉前缀 `lib`
- 使用当前 Python 解释器对应的扩展后缀
- 使用默认符号可见性，以正确导出 `PyInit_*`

### 编译与安装

- include 目录来自 Python 和 NumPy
- `lacam_online_native` 额外链接 `lacam3`
- 编译参数包含：
  - `-O3`
  - `-ffast-math`
  - `-march=native`
- 安装路径是 `${Python3_SITELIB}`
- `lacam_online_native` 构建完成后会自动复制到 `tools/extensions/`

### 自定义目标

- `build-extension`
- `test-extension`
- `clean-extension`
- `show-ext-help`

## 接口情况

### 产物

构建成功后会得到：

- `construct_features_native${PYTHON_EXT_SUFFIX}`
- `lacam_online_native${PYTHON_EXT_SUFFIX}`

### 测试接口

`test-extension` 目标通过执行：

```bash
python -c "import construct_features_native, lacam_online_native; print('✅ 扩展导入成功')"
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
