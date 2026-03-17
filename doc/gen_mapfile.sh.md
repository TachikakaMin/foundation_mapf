# `gen_mapfile.sh`

## 文件作用

这是批量地图生成脚本，用于遍历一组迷宫参数并调用 [data_generation_LACAM/maze_generator.py](/home/yimin/research/RAILGUN/data_generation_LACAM/maze_generator.py) 生成 `.map` 文件。

## 脚本流程

1. 固定目标地图尺寸为 `32 x 32`
2. 创建 `data/map_files/`
3. 遍历：
   - `density`: `0.1 -> 0.2`
   - `component`: `1 -> 10`
   - `go_straight`: `0.75 -> 0.85`
4. 用 `bc` 计算每组参数的 `num_maps`
5. 调用 `maze_generator.py` 生成地图

## 接口情况

### 依赖

- `bash`
- `bc`
- `python`
- [data_generation_LACAM/maze_generator.py](/home/yimin/research/RAILGUN/data_generation_LACAM/maze_generator.py)

### 输出

生成文件落在：

- `data/map_files/<maze-pattern>/*.map`

### 参数传递

调用 Python 脚本时传入：

- `--num_maps`
- `--width`
- `--height`
- `--obstacle_density`
- `--wall_components`
- `--go_straight`

其中 `width` 和 `height` 传的是 `30`，因为迷宫生成器内部会补边界并得到最终 `32 x 32`。

## 用法

```bash
bash gen_mapfile.sh
```
