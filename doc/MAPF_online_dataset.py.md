# `MAPF_online_dataset.py`

## 文件作用

这个文件实现了在线训练数据集 `MAPFOnlineDataset`。它不再读取预先生成的 `.mbin`，而是在 `DataLoader` worker 内直接调用原生 LACAM 扩展生成一整条轨迹，再把每个时间步展开成训练样本。

这个文件只服务训练集；验证集仍然由 [MAPF_dataset_mbin.py](/home/yimin/research/RAILGUN/MAPF_dataset_mbin.py) 读取固定离线数据。

## 主要类

### `MAPFOnlineDataset(map_files, feature_dim, feature_type, *, seed=1919180, time_limit_sec=5, verbose=0, retry_limit=64, agent_counts=(128, 96, 64, 32, 16))`

这是一个 `torch.utils.data.IterableDataset`。

参数含义：

- `map_files`: 同一尺寸的一组地图文件
- `feature_dim`: 输入特征通道数
- `feature_type`: 特征类型，通常是 `gradient`
- `seed`: 在线生成的基础随机种子
- `time_limit_sec`: 每次调用 LACAM 求解的时间上限
- `verbose`: 传给 LACAM bridge 的日志级别
- `retry_limit`: 连续生成失败时的最大重试次数
- `agent_counts`: 使用的 agent 数集合，默认沿用离线脚本中的 `128/96/64/32/16`

## 生成逻辑

### 采样分布

这个数据集按 `(map_file, agent_num)` 对进行加权采样，权重直接复用了 [gen_pathfile.sh](/home/yimin/research/RAILGUN/gen_pathfile.sh) 里的配比公式：

- `128 -> 60 + density * 2.0`
- `96 -> 40 + density * 1.0`
- `64 -> 20 + density * 0.8`
- `32 -> 5 + density * 0.1`
- `16 -> 2 + density * 0.1`

也就是说，在线训练会尽量贴近原先离线路径生成的样本分布。

### `MAPFOnlineDataset.__iter__()`

主迭代逻辑：

1. 获取 `worker_id` 和 `num_workers`
2. 为每个 worker 构造独立随机流，避免重复 scenario
3. 采样一个 `(map, agent_num)` 组合
4. 调用 `generate_lacam_solution_cpp(...)`
5. 把返回的一整条 `solution` 缓存在 worker 内
6. 将每个时间步依次转换成：
   - `feature`
   - `action`
   - `mask`
   - `file_name`

## 主要辅助接口

### `_build_pair_configs(...)`

根据地图列表和 `agent_counts` 生成所有 `(map, N)` 组合及其采样权重。

### `_make_worker_sharded_seed(...)`

构造按 worker 分片的随机种子，避免不同 worker 请求到相同 scenario。

### `_call_generator(map_file, agent_num, scenario_seed)`

调用 [tools/extensions/lacam_online_native.cpp](/home/yimin/research/RAILGUN/tools/extensions/lacam_online_native.cpp) 暴露的 `generate_lacam_solution_cpp(...)`。

当前会兼容关键字调用和几种位置参数调用方式，并在 LACAM 无解时返回 `None` 进入重试。

### `_parse_solution_output(raw)`

把 bridge 返回值标准化。当前主要契约是字典：

- `positions`: `[steps, N, 2]`
- `actions`: `[steps, N]`
- `goals`: `[N, 2]`
- `steps`
- `agent_num`

如果 `actions` 缺失，会根据相邻位置差自动推导动作编码。

### `_build_step_sample(...)`

把单个时间步转换成训练样本字典。

内部会：

1. 读取地图
2. 读取缓存距离图
3. 调用 `construct_input_feature_cpp(...)`
4. 构造离线训练同格式的：
   - `feature`
   - `action`
   - `mask`
   - `file_name`

## 接口情况

### 依赖

- [tools/extensions/__init__.py](/home/yimin/research/RAILGUN/tools/extensions/__init__.py)
- [tools/extensions/lacam_online_native.cpp](/home/yimin/research/RAILGUN/tools/extensions/lacam_online_native.cpp)
- [tools.cached_distance_reader.py](/home/yimin/research/RAILGUN/tools/cached_distance_reader.py)
- [tools.utils.py](/home/yimin/research/RAILGUN/tools/utils.py)

### 输出格式

和离线数据集保持一致：

- `feature`: `torch.FloatTensor[feature_dim, H, W]`
- `action`: `torch.LongTensor[H, W]`
- `mask`: `torch.uint8[H, W]`
- `file_name`: 一个虚拟的在线样本标识字符串

## 用法

```python
from MAPF_online_dataset import MAPFOnlineDataset

dataset = MAPFOnlineDataset(
    map_files=["data/map_files/maze-32-32-10-1-75/maze-32-32-10-1-75-0.map"],
    feature_dim=6,
    feature_type="gradient",
    time_limit_sec=5,
)
sample = next(iter(dataset))
```
