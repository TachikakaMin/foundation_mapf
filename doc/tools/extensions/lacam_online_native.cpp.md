# `tools/extensions/lacam_online_native.cpp`

## 文件作用

这个文件实现在线训练用的原生 LACAM bridge。它直接链接 vendored `lacam3` 库，在内存中创建实例、求解并返回整条轨迹，不再经过 `.path` 或 `.mbin` 的中间文件。

它是 [MAPF_online_dataset.py](/home/yimin/research/RAILGUN/MAPF_online_dataset.py) 的核心依赖。

## 主要接口

### `generate_lacam_solution_cpp(map_file, agent_num, seed, time_limit_sec=2, verbose=0)`

这是暴露给 Python 的主函数。

参数：

- `map_file`: 地图文件路径
- `agent_num`: 智能体数量
- `seed`: 用于 `Instance(map_file, agent_num, seed)` 的随机种子
- `time_limit_sec`: 求解时间上限
- `verbose`: LACAM 日志等级

返回字典：

- `positions`: `np.ndarray[int64]`, `[steps, agent_num, 2]`
- `actions`: `np.ndarray[uint8]`, `[steps, agent_num]`
- `goals`: `np.ndarray[int64]`, `[agent_num, 2]`
- `steps`
- `agent_num`

失败时会抛出 `RuntimeError`。

## 主要实现细节

### `encode_action(cur_row, cur_col, next_row, next_col)`

把相邻两步的位置变化编码成项目使用的动作编号：

- `1`
- `2`
- `3`
- `4`
- `0` 表示不动

### `generate_lacam_solution_cpp(...)` 的内部流程

1. 解析 Python 参数
2. 用 `Instance(map_file, agent_num, seed)` 创建随机 MAPF 实例
3. 创建 `Deadline`
4. 调用 `solve(ins, verbose - 1, &deadline, seed)`
5. 用 `is_feasible_solution(...)` 检查可行性
6. 分配 NumPy 数组并填充：
   - 每一步所有智能体位置
   - 每一步所有动作
   - 最终目标位置
7. 返回 Python 字典

### GIL 处理

这个扩展在执行 `solve(...)` 和可行性检查时会主动释放 Python GIL。

这意味着：

- 如果调用方在同一进程里做线程级并发，求解阶段不会被 Python GIL 卡住
- 当前正式在线训练主路径虽然主要依赖 `MAPFOnlineBufferLoader` 的多进程 producer，但这个 bridge 仍然保持线程友好，便于单进程调试和独立 profiling

## 接口情况

### 被谁调用

- [tools/extensions/__init__.py](/home/yimin/research/RAILGUN/tools/extensions/__init__.py)
- [MAPF_online_dataset.py](/home/yimin/research/RAILGUN/MAPF_online_dataset.py)
- [tools/profile_online_data.py](/home/yimin/research/RAILGUN/tools/profile_online_data.py)

### 外部依赖

- vendored `lacam3`
- Python C API
- NumPy C API

## 用法

```python
from tools.extensions import generate_lacam_solution_cpp

scenario = generate_lacam_solution_cpp(
    "data/map_files/maze-32-32-10-1-75/maze-32-32-10-1-75-0.map",
    16,
    123,
    time_limit_sec=2,
)
```
