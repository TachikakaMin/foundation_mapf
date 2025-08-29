# MAPF工具集

这个目录包含了多智能体路径规划(MAPF)相关的各种工具。

## 目录结构

```
tools/
├── README.md                    # 本文件
├── __init__.py                 # Python包初始化
├── utils.py                    # 通用工具函数
├── path_formation.py           # 路径生成相关
├── precompute_distance_maps.py # 距离地图预计算
│
├── converters/                 # 路径转换器
│   ├── cpp/                   # C++版本转换器
│   │   ├── convert_lacam_path_to_bin.cpp
│   │   ├── convert_lacam_path_to_bin (可执行文件)
│   │   ├── Makefile
│   │   └── README_CPP.md
│   └── python/                # Python版本转换器
│       └── convert_lacam_path_to_bin.py
│
├── visualization/             # 可视化工具
│   ├── visualize_path.py      # 通用路径可视化
│   ├── visualize_lacam_path.py # LACAM路径可视化
│   └── visualize_bin_path.py   # 二进制路径可视化
│
├── testing/                   # 测试工具
│   ├── test_converter.cpp     # C++测试工具
│   ├── test_converter (可执行文件)
│   └── test_integration.py    # 集成测试脚本
│
└── benchmarks/                # 性能测试
    └── benchmark.py           # 性能对比测试
```

## 快速开始

### 使用统一工具脚本（推荐）

```bash
# 首次设置 - 编译所有工具并运行测试
python mapf_tools.py setup

# 预计算距离地图（推荐先做这一步）
python mapf_tools.py distance -i /path/to/map/files/

# 转换路径文件
python mapf_tools.py convert -i /path/to/lacam/results/

# 使用Python版本转换
python mapf_tools.py convert -i /path/to/lacam/results/ -e python

# 可视化路径
python mapf_tools.py visualize /path/to/file.path

# 运行测试
python mapf_tools.py test integration
python mapf_tools.py test benchmark
```

### 直接使用各工具（高级用户）

**1. 距离地图预计算**

C++版本（推荐，高性能）:
```bash
cd converters/cpp
make                           # 编译
./precompute_distance_maps /path/to/map/files/
```

Python版本:
```bash
python -m tools.precompute_distance_maps /path/to/map/files/
```

**2. 路径转换**

C++版本（推荐，高性能）:
```bash
cd converters/cpp
make                           # 编译
./convert_lacam_path_to_bin /path/to/lacam/results/
```

Python版本:
```bash
python -m tools.converters.python.convert_lacam_path_to_bin /path/to/lacam/results/
```

**3. 验证转换结果**

```bash
cd testing
./test_converter /path/to/lacam/results/  # 验证转换正确性
python test_integration.py               # 完整集成测试
```

**4. 性能测试**

```bash
cd benchmarks
python benchmark.py  # Python vs C++性能对比
```

**5. 路径可视化**

```bash
cd visualization
python visualize_lacam_path.py /path/to/file.path
python visualize_bin_path.py /path/to/file.bin
```

## 各模块说明

### 转换器 (converters/)
- **C++版本**: 高性能，支持多线程并行处理，适合大批量转换
- **Python版本**: 易于修改和调试，适合小规模转换和开发

### 可视化 (visualization/)
- 支持多种路径格式的可视化
- 可生成静态图片或动态视频
- 支持自定义地图和路径样式

### 测试 (testing/)
- 单元测试和集成测试
- 转换结果正确性验证
- 自动化测试流程

### 基准测试 (benchmarks/)
- 性能对比分析
- 资源使用监控
- 可扩展的测试场景

## 依赖要求

### Python依赖
```bash
pip install numpy tqdm matplotlib psutil
```

### C++依赖
- C++17兼容编译器 (GCC 7+, Clang 5+)
- 支持filesystem和threading库

## 开发指南

### 添加新的转换器
1. 在`converters/`下创建新目录
2. 实现转换逻辑
3. 添加相应的测试

### 添加新的可视化工具
1. 在`visualization/`下创建新文件
2. 继承或使用现有的可视化基类
3. 更新文档

### 添加新的测试
1. 在`testing/`下添加测试文件
2. 更新集成测试脚本
3. 确保测试覆盖率

## 许可证

与项目主许可证保持一致。 