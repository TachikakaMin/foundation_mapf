# `tools/path_formation.py`

## 文件作用

这个文件实现了模型 rollout 和路径形成逻辑。给定一个模型、一个样本和步数上限，它会反复预测动作、更新所有智能体位置，并计算最终评估指标。

它被 [train.py](/home/yimin/research/RAILGUN/train.py) 和 [eval_test.py](/home/yimin/research/RAILGUN/eval_test.py) 直接调用。

## 主要函数

### `statistic_result(current_locations, goal_locations)`

计算两个基础统计量：

- 所有智能体到目标的曼哈顿距离总和
- 成功率，也就是当前位置和目标位置完全重合的智能体比例

### `move_agent(action, map_data, current_locations, temperature)`

根据动作图更新所有智能体位置，并处理冲突。

动作编码按当前实现解释为：

- `0`: 不动
- `1`: `y + 1`
- `2`: `y - 1`
- `3`: `x - 1`
- `4`: `x + 1`

冲突处理包括：

- 出界裁剪
- 多个智能体占用同一格时回退移动者
- 交换位置时同时回退

返回值：

- 新位置张量
- 原样返回的 `temperature`

### `sample_action(logits, current_locations, temperature, feature, action_choice="sample")`

把模型输出转换为实际动作图。

- `action_choice="sample"` 时，从 softmax 分布采样
- `action_choice="max"` 时，取 argmax

只有当前存在智能体的位置会保留动作，其他网格会被 `feature[1]` 生成的 mask 清零。

### `calculate_metrics(...)`

汇总 rollout 结果，返回：

- `total_cost`
- `ep_length`
- `makespan`
- `isr`
- `csr`
- `final_distance`
- `avg_density`
- `total_time`
- `throughput`

### `generate_from_possible_targets(possible_positions, position)`

从候选可通行位置里随机挑一个新目标，并保证不等于当前位置。这个函数只在 `lifelong=True` 时使用。

### `path_formation(model, val_loader, idx, device, feature_type, action_choice="sample", steps=300, log_file=None, lifelong=False)`

这是文件的核心接口。

主要流程：

1. 从 `val_loader.dataset[idx]` 取一个样本
2. 从特征图中恢复当前智能体位置和目标位置
3. 循环执行模型推理
4. 采样动作并调用 `move_agent()`
5. 用 `construct_input_feature()` 重建下一步输入
6. 在 `lifelong=True` 时为到达目标的智能体随机分配新目标
7. 累积路径、目标历史和统计量

返回值：

- `all_paths`: 每一步所有智能体的位置
- `all_goal_locations`: 每一步所有智能体的目标
- `metrics["final_distance"]`
- `file_name`

### `calculate_step_density(current_locations, map_data)`

对每个智能体计算一个 `5x5` 局部窗口内的占用密度，并返回该步所有智能体的局部密度列表。

## 接口情况

### 依赖接口

- `tools.utils.construct_input_feature(...)`
- `tools.utils.parse_file_name(file_name)`
- `tools.utils.read_distance_map(map_name)`

### 样本输入约定

`val_loader.dataset[idx]` 必须至少提供：

- `feature`
- `mask`
- `file_name`

并且特征通道约定为：

- `feature[0]`: 地图
- `feature[1]`: 智能体位置图
- `feature[2]`: 目标位置图

## 用法

```python
from tools.path_formation import path_formation

all_paths, all_goals, final_distance, file_name = path_formation(
    model=model,
    val_loader=val_loader,
    idx=0,
    device=device,
    feature_type="gradient",
    steps=300,
)
```
