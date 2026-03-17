# `data_generation_LACAM/random_generator.py`

## 文件作用

这个脚本用于生成随机障碍地图，并写成标准 `.map` 文件。

## 主要函数

### `generate_random_map(width, height, obstacle_density, seed=None)`

生成一个仅由 `.` 和 `@` 组成的地图字符串。

- `.`: 空地
- `@`: 障碍

障碍数量按 `width * height * obstacle_density` 计算，并使用 `numpy.random.default_rng(seed)` 随机放置。

### `random_map_save(width, height, obstacle_density, seed)`

调用 `generate_random_map()` 后，把结果写入：

- `data/map_files/random-<height>-<width>-<density>/<...>.map`

输出格式符合项目其它地图读取逻辑：

- `type octile`
- `height`
- `width`
- `map`
- 地图正文

## 命令行入口

支持参数：

- `--width`
- `--height`
- `--obstacle_density`
- `--num_maps`

入口会遍历 `seed in range(num_maps)` 并逐张保存地图。

## 用法

```bash
python data_generation_LACAM/random_generator.py \
  --width 32 \
  --height 32 \
  --obstacle_density 0.2 \
  --num_maps 20
```
