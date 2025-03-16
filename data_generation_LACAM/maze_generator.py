import numpy as np
import os
import argparse


def select_random_neighbor(x, y, maze_shape, rng, last_direction, go_straight):
    """
    用于从当前位置 (x, y) 中随机选择一个邻居点作为下一个扩展方向。
    函数的设计同时考虑了邻居的有效性检查和方向选择的随机性
    （倾向直线移动的概率由 go_straight 控制）
    """
    neighbor_coords = []  # 存储当前格点的有效邻居点坐标。
    probabilities = []  # 存储每个邻居被选中的概率。

    # 如果当前坐标x大于1，则可以向左扩展
    if x > 1:
        neighbor_coords.append((y, x - 2))
        probabilities.append(
            go_straight
            if (y, x - 2) == (y + last_direction[0], x + last_direction[1])
            else (1 - go_straight)
        )
    # 如果当前坐标x小于迷宫的宽度减2，则可以向右扩展
    if x < maze_shape[1] - 2:
        neighbor_coords.append((y, x + 2))
        probabilities.append(
            go_straight
            if (y, x + 2) == (y + last_direction[0], x + last_direction[1])
            else (1 - go_straight)
        )
    # 如果当前坐标y大于1，则可以向上扩展
    if y > 1:
        neighbor_coords.append((y - 2, x))
        probabilities.append(
            go_straight
            if (y - 2, x) == (y + last_direction[0], x + last_direction[1])
            else (1 - go_straight)
        )
    # 如果当前坐标y小于迷宫的高度减2，则可以向下扩展
    if y < maze_shape[0] - 2:
        neighbor_coords.append((y + 2, x))
        probabilities.append(
            go_straight
            if (y + 2, x) == (y + last_direction[0], x + last_direction[1])
            else (1 - go_straight)
        )

    # 检查是否有可用邻居
    if not neighbor_coords:
        return None, None, last_direction

    # 调整概率分布
    total = sum(probabilities)
    probabilities = [prob / total for prob in probabilities]

    # 根据概率分布随机选择邻居
    chosen_index = rng.choice(range(len(neighbor_coords)), p=probabilities)
    next_y, next_x = neighbor_coords[chosen_index]
    new_direction = (next_y - y, next_x - x)
    return next_x, next_y, new_direction


def generate_maze(
    width, height, obstacle_density, wall_components, go_straight, seed=42
):
    height = height // 2 * 2 + 2
    width = width // 2 * 2 + 2
    # 准备文件路径
    output_dir = "data/map_files/"
    maze_dir_name = f"maze-{height}-{width}-{int(obstacle_density*100)}-{wall_components}-{int(go_straight*100)}"
    output_map_dir = os.path.join(output_dir, maze_dir_name)
    output_path = os.path.join(output_map_dir, f"{maze_dir_name}-{seed}.map")

    # 检查迷宫是否已存在
    if os.path.exists(output_path):
        print(f"Maze with seed {seed} already exists, skipping generation.")
        return

    rng = np.random.default_rng(seed)
    maze_shape = (
        height + 1,
        width + 1,
    )  # 确保迷宫的矩阵尺寸为奇数，并在外围留出一圈墙；墙之后会被删除
    # 根据迷宫的面积、障碍物密度和墙体单元数，计算墙体的数量。
    density = int(maze_shape[0] * maze_shape[1] * obstacle_density // wall_components)

    maze_grid = np.zeros(maze_shape, dtype="int")
    # 第一行和最后一行设置为墙
    maze_grid[0, :] = 1
    maze_grid[-1, :] = 1
    # 第一列和最后一列设置为墙
    maze_grid[:, 0] = 1
    maze_grid[:, -1] = 1

    # 随机选择墙体的起始点
    for i in range(density):
        # 确保起点的坐标为偶数，这样墙和路径交替分布。
        x = rng.integers(0, maze_shape[1] // 2) * 2
        y = rng.integers(0, maze_shape[0] // 2) * 2
        maze_grid[y, x] = 1

        # 随机扩展墙体
        last_direction = (0, 0)  # 初始方向为无，因为是起始点
        for j in range(wall_components):
            next_x, next_y, last_direction = select_random_neighbor(
                x, y, maze_shape, rng, last_direction, go_straight
            )
            if (
                next_x is not None and maze_grid[next_y, next_x] == 0
            ):  # 检查选择的邻居是否为空地
                maze_grid[next_y, next_x] = 1
                maze_grid[next_y + (y - next_y) // 2, next_x + (x - next_x) // 2] = (
                    1  # 连接当前点和邻居点之间的路径（中间点），将其也设为 1
                )
                x, y = next_x, next_y

    maze_grid = maze_grid[:-1, :-1]  # 删除一半外围的墙，因为想要shape是偶数

    maze_str = []
    for line in maze_grid:
        maze_str.append("".join(["@" if x == 1 else "." for x in line]))
    maze_str = "\n".join(["".join(row) for row in maze_str])

    # 保存迷宫
    os.makedirs(output_map_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("type octile\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write("map\n")
        f.write(maze_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", "-w", type=int, default=30)
    parser.add_argument("--height", "-he", type=int, default=30)
    parser.add_argument("--obstacle_density", "-d", type=float, default=0.2)
    parser.add_argument("--wall_components", "-c", type=int, default=4)
    parser.add_argument("--go_straight", "-g", type=float, default=0.75)
    parser.add_argument("--num_maps", "-n", type=int, default=20)
    args = parser.parse_args()

    for seed in range(args.num_maps):
        generate_maze(
            args.width,
            args.height,
            args.obstacle_density,
            args.wall_components,
            args.go_straight,
            seed,
        )
