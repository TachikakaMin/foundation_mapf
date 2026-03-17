# `tools/precompute_distance_maps.py`

## 文件作用

这个脚本用于批量预计算地图距离图，并把结果保存为 `pickle` 格式的 `.pkl` 文件。

它是数据准备阶段的离线工具。

## 主要函数

### `process_single_map(map_file)`

处理一张地图：

1. 把 `.map` 路径映射成 `distance_maps/*.pkl`
2. 如果输出已存在则跳过
3. 调用 `read_map()` 读取地图
4. 调用 `create_distance_map()` 计算全图距离
5. 写入 `pickle`

返回值：

- 成功处理时返回原始 `map_file`
- 已存在时返回 `None`

### `main()`

命令行入口。

行为：

- 要求参数个数为 1：`<map_files_dir>`
- 递归搜索目录内全部 `.map`
- 使用 `ProcessPoolExecutor` 并行调用 `process_single_map()`
- 统计成功处理数量

## 接口情况

### 依赖

- [tools/utils.py](/home/yimin/research/RAILGUN/tools/utils.py) 的：
  - `create_distance_map()`
  - `read_map()`

### 输出路径规则

输出路径通过字符串替换得到：

- `map_files/.../*.map`
- `distance_maps/.../*.pkl`

## 用法

```bash
python -m tools.precompute_distance_maps data/map_files
```
