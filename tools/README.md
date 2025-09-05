# MAPF工具集 v2.0

经过性能优化和重构的多智能体路径规划(MAPF)工具集。

## 🏗️ 目录结构

```
tools/
├── README.md                    # 本文件
├── __init__.py                 # 工具集主入口
│
├── core/                       # 🔧 核心工具
│   ├── utils.py               # 基础工具函数
│   ├── path_formation.py      # 路径生成
│   ├── mapf_tools.py          # 统一命令行工具
│   ├── status_check.py        # 状态检查
│   └── example_usage.py       # 使用示例
│
├── data_processing/           # 📊 数据处理
│   ├── cached_distance_reader.py    # 缓存距离地图读取器
│   ├── distance_map_reader.py       # 距离地图读取器
│   └── precompute_distance_maps.py  # 距离预计算
│
├── extensions/                # ⚡ C++扩展
│   ├── construct_features_native.cpp  # 特征构建C++实现
│   ├── construct_features_native.so   # 编译后的扩展
│   └── setup_native.py              # 编译脚本
│
├── converters/                # 🔄 数据转换器
│   ├── cpp/                   # C++转换器
│   │   ├── convert_lacam_path_to_bin.cpp
│   │   ├── precompute_distance_maps.cpp
│   │   └── Makefile
│   └── python/                # Python转换器
│       └── convert_lacam_path_to_bin.py
│
├── benchmarks/                # 📈 性能测试
│   ├── cpp_feature_benchmark.py     # C++特征构建性能测试
│   ├── distance_benchmark.py        # 距离计算性能测试
│   ├── training_benchmark.py        # 训练性能测试
│   └── quick_training_test.py       # 快速训练测试
│
├── visualization/             # 📊 可视化工具
│   ├── visualize_path.py      # 通用路径可视化
│   ├── visualize_lacam_path.py # LACAM路径可视化
│   └── visualize_bin_path.py   # 二进制路径可视化
│
└── testing/                   # 🧪 测试工具
    ├── test_converter.cpp     # C++测试工具
    ├── test_converter         # 编译后的测试工具
    └── test_integration.py    # 集成测试脚本
```

## 🚀 快速开始

### 1. 一键设置和转换

```bash
# 在tools目录中
cd foundation_mapf/tools

# 使用CMake构建所有工具
./build.sh

# 或者手动使用CMake
mkdir build && cd build
cmake ..
make -j$(nproc)

# 预计算距离地图
python core/mapf_tools.py distance -i ../data/map_files

# 转换路径文件为高效的.mbin格式
python core/mapf_tools.py convert -i ../data/path_files

# 检查状态
python core/status_check.py
```

### 2. 开始训练

```bash
# 回到项目根目录
cd ../

# 开始高效训练（使用优化后的数据）
python train.py --batch_size 64 --epochs 100
```

## 📊 性能优化成果

### ✅ 已实现的优化：

1. **文件合并优化**
   - 从587,840个小文件 → 10,515个合并文件
   - 数据大小从30.2GB → 12GB (60%减少)
   - 文件数量减少55倍

2. **C++距离地图预计算**
   - 2,106个地图文件，9秒完成 (vs Python版本几分钟)
   - 5-10倍性能提升

3. **距离地图缓存**
   - 避免重复加载相同地图
   - 内存高效的缓存策略

4. **数据加载优化**
   - 从125ms/样本 → 12.68ms/样本 (10倍提升)
   - 吞吐量从8样本/秒 → 78.8样本/秒

5. **小地图过滤**
   - 自动过滤<32x32的地图，避免UNet尺寸问题

## 🔧 工具使用

### 统一命令行工具

```bash
# 查看所有可用命令
python core/mapf_tools.py --help

# 数据转换
python core/mapf_tools.py convert -i ../data/path_files

# 距离预计算
python core/mapf_tools.py distance -i ../data/map_files

# 运行测试
python core/mapf_tools.py test integration

# 性能测试
python core/mapf_tools.py test benchmark
```

### 性能测试

```bash
# C++特征构建性能测试
python benchmarks/cpp_feature_benchmark.py

# 端到端训练性能测试
python benchmarks/quick_training_test.py

# 距离计算性能测试
python benchmarks/distance_benchmark.py
```

### 可视化

```bash
# 可视化路径文件
python visualization/visualize_lacam_path.py /path/to/file.path

# 可视化二进制文件
python visualization/visualize_bin_path.py /path/to/file.mbin
```

## 📈 性能基准

基于实际测试的性能数据：

| 组件 | 原始性能 | 优化后性能 | 提升倍数 |
|------|----------|------------|----------|
| 文件数量 | 587,840个 | 10,515个 | 55x减少 |
| 数据大小 | 30.2GB | 12GB | 2.5x减少 |
| 距离预计算 | 几分钟 | 9秒 | 10x+ 加速 |
| 数据加载 | 125ms/样本 | 12.68ms/样本 | 10x 加速 |
| 特征构建 | 0.28ms | 0.01ms | 28x 加速 |

## 🔄 数据流程

```
原始数据(.path) 
    ↓ [C++转换器，24线程并行]
合并数据(.mbin, 55倍文件减少)
    ↓ [缓存距离地图]
高效训练数据
    ↓ [优化的DataLoader]
快速训练 (10倍数据加载加速)
```

## 🎯 使用建议

1. **首次使用**: 运行`python core/mapf_tools.py setup`进行一键设置
2. **大数据集**: 使用C++转换器和预计算工具
3. **开发调试**: 使用Python工具进行快速迭代
4. **性能监控**: 定期运行benchmark测试

## 📝 更新日志

### v2.0.0
- 重新组织目录结构
- 添加C++扩展支持
- 实现文件合并优化
- 添加缓存机制
- 10倍数据加载性能提升

### v1.0.0
- 基础工具集
- Python实现
- 基本的数据转换功能

## 许可证

与项目主许可证保持一致。 