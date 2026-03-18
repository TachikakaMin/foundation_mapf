# `gen_online_testset.sh`

## 文件作用

这是给在线训练准备“小规模固定验证 / 测试集”的脚本。

它不会生成全量离线训练数据，而是：

- 从已有 `data/map_files` 中选一小部分地图
- 只为这些地图生成少量 `.path`
- 把这些 `.path` 转成固定离线 `.mbin`
- 输出到独立目录 `data/online_eval_input_data`

这个脚本主要服务 [README.md](/home/yimin/research/RAILGUN/README.md) 里的 online workflow。

## 脚本流程

1. 检查 `parallel` 和 `data_generation_LACAM/lacam3/build/main`
2. 从 `MAP_GLOB` 中选取最多 `MAX_MAPS_TOTAL` 张地图
3. 只为这些地图补距离图
4. 按 `AGENT_COUNTS` 和 `SEEDS_PER_AGENT` 生成少量 `.path`
5. 把 `.path` 转成 `.mbin`
6. 写到 `OUTPUT_ROOT`

## 默认配置

脚本通过环境变量控制规模，默认值是：

- `MAP_GLOB=data/map_files/maze-*/*.map`
- `MAX_MAPS_TOTAL=8`
- `AGENT_COUNTS="128 96 64 32 16"`
- `SEEDS_PER_AGENT=1`
- `PARALLEL_JOBS=$(nproc)`
- `PARALLEL_TIMEOUT_SEC=5`
- `PATH_ROOT=data/online_eval_path_files`
- `OUTPUT_ROOT=data/online_eval_input_data`

## 输出

### 路径文件

- `data/online_eval_path_files/<map_pattern>/<map_name>-<N>/<map_name>-<N>-<seed>.path`

### 验证 / 测试 `.mbin`

- `data/online_eval_input_data/<map_pattern>/<map_name>-<N>.mbin`

### 距离图

仍然写到项目统一路径：

- `data/distance_maps/...`

## 接口情况

### 依赖

- `bash`
- `GNU parallel`
- `python`
- `nproc`
- [tools/precompute_distance_maps.py](/home/yimin/research/RAILGUN/tools/precompute_distance_maps.py)
- [tools/convert_lacam_path_to_bin.py](/home/yimin/research/RAILGUN/tools/convert_lacam_path_to_bin.py)
- `data_generation_LACAM/lacam3/build/main`

### 适用场景

- online 训练前准备固定离线验证集
- online 训练前准备固定 inference sample
- 不想生成完整离线训练 `.mbin`

## 用法

默认用法：

```bash
bash gen_online_testset.sh
```

调小或调大测试集规模：

```bash
MAX_MAPS_TOTAL=8 SEEDS_PER_AGENT=1 bash gen_online_testset.sh
MAX_MAPS_TOTAL=16 AGENT_COUNTS="64 32 16" SEEDS_PER_AGENT=2 bash gen_online_testset.sh
```
