# `tools/visualize_bin_path.py`

## 文件作用

这个脚本用于把旧格式 `.bin` 轨迹文件直接解析成位置序列，并调用 [tools/visualize_path.py](/home/yimin/research/RAILGUN/tools/visualize_path.py) 做可视化。

## 主要函数

### `parse_bin_path(bin_file)`

解析 `.bin` 文件中的轨迹。

读取顺序是：

- `steps`
- `agent_num`
- 对每个时间步：
  - 读取所有智能体位置
  - 读取所有智能体动作

返回值：

- `all_paths`，长度为 `steps`，每个元素是当前时刻所有智能体坐标

说明：

- 当前函数会把动作字段读出来但不使用，只保留位置序列。

## 命令行入口

脚本入口要求：

```bash
python -m tools.visualize_bin_path <path_to_bin_file>
```

入口逻辑会把最后一帧位置重复为所有时间步的目标位置，然后调用 `visualize_path(..., show=True)`。

## 接口情况

- 输入：单个 `.bin` 文件
- 输出：交互式可视化窗口

## 用法

```bash
python -m tools.visualize_bin_path data/input_data/example.bin
```
