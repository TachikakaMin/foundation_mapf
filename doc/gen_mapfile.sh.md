# `gen_mapfile.sh`

## 文件作用

这是批量地图生成脚本，用于遍历一组迷宫参数并调用 [data_generation_LACAM/maze_generator.py](/home/yimin/research/RAILGUN/data_generation_LACAM/maze_generator.py) 生成 `.map` 文件。

当前版本支持：

- GNU `parallel` 并行生成
- `parallel --bar` 进度条
- 通过环境变量覆盖并行度和参数范围

## 脚本流程

1. 检查 `parallel` 和 `bc`
2. 读取环境变量或默认值：
   - `HEIGHT=32`
   - `WIDTH=32`
   - `PARALLEL_JOBS=$(nproc)`
3. 创建 `data/map_files/`
4. 遍历：
   - `density`: `0.1 -> 0.2`
   - `component`: `1 -> 10`
   - `go_straight`: `0.75 -> 0.85`
5. 用 `bc` 计算每组参数的 `num_maps`
6. 把每组任务写入临时任务表
7. 用 GNU `parallel` 并行调用 `maze_generator.py`

## 接口情况

### 依赖

- `bash`
- `GNU parallel`
- `bc`
- `python`
- [data_generation_LACAM/maze_generator.py](/home/yimin/research/RAILGUN/data_generation_LACAM/maze_generator.py)

### 输出

生成文件落在：

- `data/map_files/<maze-pattern>/*.map`

### 参数传递

并行任务最终调用 Python 脚本时传入：

- `--num_maps`
- `--width`
- `--height`
- `--obstacle_density`
- `--wall_components`
- `--go_straight`

其中 `width` 和 `height` 传的是 `30`，因为迷宫生成器内部会补边界并得到最终 `32 x 32`。

## 可调环境变量

- `HEIGHT`
- `WIDTH`
- `DENSITY_START`
- `DENSITY_END`
- `DENSITY_STEP`
- `COMPONENT_START`
- `COMPONENT_END`
- `GO_STRAIGHT_START`
- `GO_STRAIGHT_END`
- `GO_STRAIGHT_STEP`
- `PARALLEL_JOBS`

## 用法

```bash
bash gen_mapfile.sh

# 指定并行度
PARALLEL_JOBS=8 bash gen_mapfile.sh
```
