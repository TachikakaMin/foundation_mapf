# `MAPF_online_dataset.py`

## 文件作用

这个文件实现在线训练的数据生产链路，职责分成两层：

- `MAPFOnlineDataset`：负责采样 `(map_file, agent_num)`、调用原生 LACAM bridge、把一整条轨迹展开成逐 step 训练样本。
- `MAPFOnlineBufferLoader`：负责在线训练时真正使用的“全局 step buffer”。它用多进程持续生成 scenario，再由主进程线程把 scenario 展开成 step 样本并组 batch。

验证集仍然是固定离线 `.mbin`，不走这个文件。

## 主要类与函数

### `MAPFOnlineDataset(...)`

这是一个 `torch.utils.data.IterableDataset`，更偏向“scenario -> step sample”的基础组件。

构造参数包括：

- `map_files`: 同一尺寸的一组地图文件
- `feature_dim`: 输入特征通道数
- `feature_type`: 特征类型，通常是 `gradient`
- `seed`: 在线生成的基础随机种子
- `time_limit_sec`: 每次调用 LACAM 的时间上限
- `verbose`: 传给 bridge 的日志级别
- `retry_limit`: 单条 scenario 的最大重试次数
- `agent_counts`: 使用的 agent 数集合，默认沿用离线脚本的 `128/96/64/32/16`
- `buffer_size` / `buffer_workers` / `buffer_timeout_sec`: 仅在直接迭代 `MAPFOnlineDataset` 时启用其内部线程缓冲；正式训练主路径通常不用这套内部缓冲，而是交给 `MAPFOnlineBufferLoader`

### `_scenario_producer_process(...)`

这是在线训练主路径里的独立子进程入口。

它只做重活：

- 随机采样 `(map, agent_num)`
- 调用 `generate_lacam_solution_cpp(...)`
- 把原始 `positions/actions/goals` 放进进程间 `scenario_queue`

它不做 feature construction，这样可以让 feature 构造集中在主进程，减少重复缓存和额外同步。

### `MAPFOnlineBufferLoader(...)`

这是 `train.py` 在线训练时真正创建的 loader。

它不是 `torch.utils.data.DataLoader`，而是一个自定义迭代器，内部结构是：

1. `buffer_workers` 个 producer 进程持续做 LACAM 求解
2. 一个主进程 builder 线程从 `scenario_queue` 取出轨迹
3. builder 把轨迹展开成 step 样本并写入 `_step_queue`
4. 训练循环持续从 `_step_queue` 取样本，攒够 `batch_size` 后返回一个 batch

## 采样与种子逻辑

### `(map_file, agent_num)` 采样分布

这个文件按 `(map_file, agent_num)` 对进行加权采样，权重复用了 [gen_pathfile.sh](/home/yimin/research/RAILGUN/gen_pathfile.sh) 的配比公式：

- `128 -> 60 + density * 2.0`
- `96 -> 40 + density * 1.0`
- `64 -> 20 + density * 0.8`
- `32 -> 5 + density * 0.1`
- `16 -> 2 + density * 0.1`

也就是说，在线训练尽量贴近原先离线路径生成的样本分布。

### `_make_worker_sharded_seed(...)`

这个辅助函数会把：

- 基础 `seed`
- pair 专属偏移
- pair 内部计数器
- `worker_id`
- `num_workers`

合并成最终 scenario seed，避免不同 worker 重复请求到同一个 scenario。

## 两种迭代模式

### `MAPFOnlineDataset.__iter__()`

这个基础 dataset 自身支持两种模式：

- `buffer_size == 0`：走 `_iter_simple()`，单线程消费当前 scenario，并用一个 `ThreadPoolExecutor(max_workers=1)` 预取下一条 scenario
- `buffer_size > 0`：走 `_iter_buffered()`，用多个线程把 scenario 放进本地 `queue.Queue`

这层能力主要用于组件级复用和调试。

### `MAPFOnlineBufferLoader.__iter__()`

正式训练在线模式使用的是这层：

- 先确保 producer 进程和 builder 线程已启动
- 反复从 `_step_queue` 取 step 样本
- 累积到 `batch_size` 后用 `_collate_step_samples(...)` 拼成：
  - `feature`
  - `action`
  - `mask`
  - `file_name`

## buffer 语义

### `online_buffer_size`

当前语义是“step 数”，不是 trajectory 数。

更准确地说：

- 一个 `MAPFOnlineBufferLoader` 对应一个地图尺寸分组
- `online_buffer_size` 是该尺寸分组内 `_step_queue` 的容量
- 队列里存的是展开后的逐 step 样本

所以它的单位应理解为：`steps per map-size group`。

### `num_workers` 与在线 buffer 的关系

当前代码里已经统一成一个概念：

- 在线训练时，`train.py` 会把 `args.num_workers` 传给 `MAPFOnlineBufferLoader(buffer_workers=...)`
- 也就是说，在线模式下的 `num_workers` 代表 scenario producer 进程数
- 同一个 `num_workers` 也复用于离线验证集和 inference sample 的 `DataLoader`

现在已经没有单独的 `online_buffer_workers` 配置。

## 主要辅助接口

### `_call_generator(map_file, agent_num, scenario_seed)`

调用 [tools/extensions/lacam_online_native.cpp](/home/yimin/research/RAILGUN/tools/extensions/lacam_online_native.cpp) 暴露的 `generate_lacam_solution_cpp(...)`。

当前会兼容关键字调用和几种位置参数调用方式；LACAM 无解时返回 `None`，交给上层重试。

### `_parse_solution_output(raw)` / `_normalize_solution_arrays(...)`

把 bridge 返回值标准化成统一 numpy 结构：

- `positions`: `[steps, N, 2]`
- `actions`: `[steps, N]`
- `goals`: `[N, 2]`

如果 `actions` 缺失，会根据相邻位置差自动推导。

### `_build_step_sample(...)`

把单个时间步转换成训练样本字典。

内部会：

1. 读取地图并缓存
2. 读取距离图并缓存
3. 调用 `construct_input_feature_cpp(...)`
4. 构造训练样本：
   - `feature`
   - `action`
   - `mask`
   - `file_name`

### `_builder_loop()`

这是 `MAPFOnlineBufferLoader` 的核心 builder 线程：

- 从进程间 `scenario_queue` 读取 raw scenario
- 调用 `MAPFOnlineDataset` 的标准化逻辑
- 把一条轨迹拆成所有 step 样本
- 将 step 样本压入 `_step_queue`

这样 producer 进程专注求解，主进程专注特征构造和组 batch。

## 关闭与清理

### `MAPFOnlineBufferLoader.close()`

在线 loader 需要显式关闭。

当前关闭逻辑会：

- 设置 stop event
- `cancel_join_thread()`，避免 `mp.Queue` 在解释器退出时卡住
- kill / join 所有 producer 进程
- 排空 `_step_queue`，帮助 builder 线程退出
- join builder 线程

因此 `train.py` 会在 `finally` 中调用 `close_loader_collection(...)` 做清理。

## 输出格式

在线训练最终返回的 batch 与离线 `.mbin` 数据格式保持一致：

- `feature`: `torch.FloatTensor[B, C, H, W]`
- `action`: `torch.LongTensor[B, H, W]`
- `mask`: `torch.uint8[B, H, W]`
- `file_name`: `list[str]`

## 依赖

- [tools/extensions/__init__.py](/home/yimin/research/RAILGUN/tools/extensions/__init__.py)
- [tools/extensions/lacam_online_native.cpp](/home/yimin/research/RAILGUN/tools/extensions/lacam_online_native.cpp)
- [tools.cached_distance_reader.py](/home/yimin/research/RAILGUN/tools/cached_distance_reader.py)
- [tools.utils.py](/home/yimin/research/RAILGUN/tools/utils.py)

## 用法

直接测试单个在线 batch：

```python
from MAPF_online_dataset import MAPFOnlineBufferLoader

loader = MAPFOnlineBufferLoader(
    map_files=["data/map_files/maze-32-32-10-1-75/maze-32-32-10-1-75-0.map"],
    feature_dim=6,
    feature_type="gradient",
    batch_size=64,
    buffer_size=1024,
    buffer_workers=4,
)

batch = next(iter(loader))
print(batch["feature"].shape)
loader.close()
```
