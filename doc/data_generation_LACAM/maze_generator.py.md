# `data_generation_LACAM/maze_generator.py`

## 文件作用

这个脚本用于生成带有连通墙结构的迷宫地图，是项目数据集生成流程里最常用的地图生成器。

## 主要函数

### `select_random_neighbor(x, y, maze_shape, rng, last_direction, go_straight)`

从当前墙体端点选择下一个扩展方向。

特点：

- 只在合法边界内选择
- 会根据 `go_straight` 对“保持当前方向”做概率偏置

返回：

- `next_x`
- `next_y`
- `new_direction`

### `generate_maze(width, height, obstacle_density, wall_components, go_straight, seed=42)`

生成并保存单张迷宫。

主要流程：

1. 把输入宽高调整为偶数边界友好的尺寸
2. 根据障碍密度和 `wall_components` 计算墙体数量
3. 初始化带外边框的网格
4. 随机选取若干墙体种子
5. 反复调用 `select_random_neighbor()` 扩展墙体
6. 去掉多余外边框
7. 转成 `.map` 文本并写盘

若目标文件已存在，会直接跳过。

## 输出路径

生成结果保存在：

- `data/map_files/maze-<height>-<width>-<density>-<wall_components>-<go_straight>/<...>.map`

## 命令行入口

支持参数：

- `--width`
- `--height`
- `--obstacle_density`
- `--wall_components`
- `--go_straight`
- `--num_maps`

主程序会按不同 `seed` 循环调用 `generate_maze()`。

## 用法

```bash
python data_generation_LACAM/maze_generator.py \
  --width 30 \
  --height 30 \
  --obstacle_density 0.2 \
  --wall_components 4 \
  --go_straight 0.75 \
  --num_maps 20
```
