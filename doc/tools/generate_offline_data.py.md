# `tools/generate_offline_data.py`

## 文件作用

这个脚本用于并行生成离线训练数据，直接调用 LACAM C++ 求解器生成 scenario 并写入 `.mbin` 格式。

相比在线训练模式（实时生成数据），预生成离线数据的优势：

- ✅ 训练速度更快（无需等待 LACAM 求解）
- ✅ 可以提前验证数据质量和分布
- ✅ 支持多次训练复用同一数据集
- ✅ 便于数据集版本管理和共享

## 核心功能

### 1. 多进程并行生成

- 使用 `multiprocessing` 启动多个 worker 进程
- 每个 worker 独立调用 LACAM 求解器
- 通过队列收集结果并写入文件
- 支持自定义并行度（`--workers`）

### 2. 智能采样分布

采样权重函数 `offline_num_paths_weight(density, agent_num)` 与 `MAPF_online_dataset.py` 保持一致，并扩展支持 512 和 1024 agents：

```python
if agent_num == 1024:
    value = 5 + d * 0.2      # 低权重（难求解）
elif agent_num == 512:
    value = 10 + d * 0.3
elif agent_num == 256:
    value = 20 + d * 0.5
elif agent_num == 128:
    value = 60 + d * 2.0     # 最高权重
elif agent_num == 96:
    value = 40 + d * 1.0
# ...
```

这确保了离线数据的分布与在线训练一致。

### 3. .mbin 文件格式

生成的 `.mbin` 文件格式与 `MAPF_dataset_mbin.py` 兼容：

**文件结构**：
```
[文件头 16 bytes]
  - num_scenarios (4 bytes)
  - padding (12 bytes)

[索引表 num_scenarios × 272 bytes]
  每个索引条目：
  - offset (8 bytes)
  - data_size (4 bytes)
  - steps (2 bytes)
  - agent_num (2 bytes)
  - file_name (256 bytes)

[数据区]
  每个 scenario：
  - steps (2 bytes)
  - agent_num (2 bytes)
  - 每个 timestep:
    - positions: agent_num × 2 bytes (x, y)
    - actions: agent_num × 1 byte
```

### 4. 自动分组写入

- 按 `(map_name, agent_num)` 分组
- 每组达到 `--scenarios_per_file` 数量时自动写入一个 `.mbin` 文件
- 避免单文件过大，便于管理和加载

## 主要函数

### `build_pair_configs(map_files, agent_counts)`

构建 (地图, agent数量) 配置对列表，并计算每个配置对的采样权重。

### `generate_one_scenario(pair, scenario_seed, time_limit_sec, verbose)`

生成单个 scenario：
1. 调用 `generate_lacam_solution_cpp()` 求解
2. 解析输出为 `(positions, actions, goals)`
3. 序列化为 `.mbin` 格式的二进制数据

### `worker_process(...)`

Worker 进程主循环：
- 按采样权重随机选择配置对
- 生成 scenario 并发送到结果队列
- 支持失败重试（`--retry_limit`）

### `write_mbin_file(output_path, scenarios)`

将 scenario 列表写入 `.mbin` 文件，包括：
- 文件头
- 索引表
- 数据区

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config.online.yaml` | 配置文件路径（用于加载地图路径等） |
| `--output_dir` | `data/input_data` | 输出目录 |
| `--num_scenarios` | `100000` | 生成的 scenario 总数 |
| `--workers` | `20` | 并行 worker 数量 |
| `--time_limit` | `5` | LACAM 求解时间限制（秒） |
| `--retry_limit` | `3` | 每个 scenario 的重试次数 |
| `--seed` | `1919180` | 随机种子 |
| `--agent_counts` | `[16,32,64,96,128,256,512,1024]` | Agent 数量列表 |
| `--map_pattern` | `data/map_files/maze-*/*.map` | 地图文件匹配模式 |
| `--scenarios_per_file` | `1000` | 每个 .mbin 文件包含的 scenario 数量 |

## 输出

生成的文件结构：

```
data/input_data/
├── maze-32-32-10-1-75-128/
│   ├── maze-32-32-10-1-75-128-1000.mbin
│   ├── maze-32-32-10-1-75-128-2000.mbin
│   └── maze-32-32-10-1-75-128-final.mbin
├── maze-64-64-10-1-75-256/
│   └── ...
└── maze-64-64-10-1-75-1024/
    └── ...
```

每个 `.mbin` 文件包含多个 scenario，每个 scenario 包含完整的轨迹数据（所有 timestep）。

## 性能估算

基于测试结果：

- 单进程速率：~70 scenarios/s
- 20 进程并行：~1400 scenarios/s
- 生成 10 万个 scenario：约 **7-15 分钟**
  - 小 agent 数量（16-128）：快
  - 大 agent 数量（512-1024）：慢

存储空间估算：

- 每个 scenario：~50 steps × agent_num × 3 bytes
- 128 agents × 50 steps = ~19 KB/scenario
- 1024 agents × 50 steps = ~150 KB/scenario
- 10 万个 scenario（混合分布）：约 **5-10 GB**

## 依赖

- `tools.extensions.generate_lacam_solution_cpp`：LACAM C++ 求解器
- `numpy`：数组处理
- `multiprocessing`：并行生成

**重要**：需要使用与 C++ 扩展编译版本匹配的 Python 解释器（通常是 Python 3.11）。

## 用法

### 基础用法

生成 10 万个 scenario，使用 20 个并行进程：

```bash
python tools/generate_offline_data.py \
    --output_dir data/input_data \
    --num_scenarios 100000 \
    --workers 20
```

### 自定义配置

指定地图和 agent 数量分布：

```bash
python tools/generate_offline_data.py \
    --output_dir data/input_data \
    --num_scenarios 100000 \
    --workers 20 \
    --time_limit 5 \
    --retry_limit 5 \
    --agent_counts 16 32 64 96 128 256 512 1024 \
    --map_pattern "data/map_files/maze-*/*.map" \
    --scenarios_per_file 500
```

### 只生成特定地图

只使用 32x32 地图：

```bash
python tools/generate_offline_data.py \
    --output_dir data/input_data_32x32 \
    --num_scenarios 50000 \
    --workers 20 \
    --map_pattern "data/map_files/maze-32-32-*/*.map" \
    --agent_counts 64 96 128
```

### 小规模测试

快速测试脚本是否正常工作：

```bash
python tools/generate_offline_data.py \
    --output_dir data/input_data_test \
    --num_scenarios 100 \
    --workers 4 \
    --map_pattern "data/map_files/maze-32-32-10-1-75/*.map" \
    --agent_counts 64 128
```

## 数据量规划

根据训练需求规划数据量：

| 训练步数 | Batch Size | 总样本数 | Scenario 数量（约） | 预计生成时间（20 workers） |
|---------|-----------|---------|-------------------|------------------------|
| 50k | 64 | 320 万 | 4 万 | 3-5 分钟 |
| 100k | 64 | 640 万 | 8 万 | 7-15 分钟 |
| 200k | 64 | 1280 万 | 16 万 | 15-30 分钟 |
| 500k | 64 | 3200 万 | 40 万 | 40-80 分钟 |

**注意**：每个 scenario 平均产生 ~50-100 个 timestep 样本。

## 与在线训练的对比

| 特性 | 离线数据（本脚本） | 在线训练 |
|------|------------------|---------|
| 训练速度 | ✅ 快（无 LACAM 开销） | ⚠️ 慢（实时求解） |
| 存储需求 | ⚠️ 需要 5-20 GB | ✅ 无需存储 |
| 数据分布 | ⚠️ 固定（需提前规划） | ✅ 灵活（可动态调整） |
| 复用性 | ✅ 可多次训练 | ❌ 每次重新生成 |
| 调试便利 | ✅ 可检查数据质量 | ⚠️ 难以检查 |

## 最佳实践

1. **先小规模测试**：用 `--num_scenarios 100` 验证脚本和参数
2. **监控失败率**：如果失败率 > 10%，考虑增加 `--time_limit` 或 `--retry_limit`
3. **平衡分布**：确保 `--agent_counts` 覆盖训练和测试需要的所有密度
4. **分批生成**：大数据集可以分多次生成，避免单次运行时间过长
5. **验证格式**：生成后用 `MAPF_dataset_mbin.py` 加载验证格式正确性

## 故障排查

### 问题：ImportError: No module named 'tools.extensions.construct_features_native'

**原因**：C++ 扩展未编译或 Python 版本不匹配

**解决**：
```bash
# 检查 Python 版本
python --version

# 重新编译 C++ 扩展
cd tools
bash build.sh build
cd ..
```

### 问题：生成速率很慢（< 10 scenarios/s）

**原因**：
- `--time_limit` 太大
- 地图太大或 agent 数量太多
- worker 数量不足

**解决**：
- 减小 `--time_limit` 到 3-5 秒
- 增加 `--workers` 到 CPU 核心数
- 分开生成不同难度的数据

### 问题：大量 scenario 生成失败

**原因**：
- 地图太小，agent 数量太多（无解）
- `--time_limit` 太小

**解决**：
- 检查 `--agent_counts` 是否合理
- 增加 `--time_limit` 到 10 秒
- 增加 `--retry_limit` 到 10

## 相关文件

- [MAPF_online_dataset.py](/home/yimintan/research/RAILGUN/MAPF_online_dataset.py)：在线数据生成（采样权重逻辑）
- [MAPF_dataset_mbin.py](/home/yimintan/research/RAILGUN/MAPF_dataset_mbin.py)：.mbin 文件读取
- [tools/extensions/lacam_online_native.cpp](/home/yimintan/research/RAILGUN/tools/extensions/lacam_online_native.cpp)：LACAM 求解器
- [config.offline.yaml](/home/yimintan/research/RAILGUN/config.offline.yaml)：离线训练配置
