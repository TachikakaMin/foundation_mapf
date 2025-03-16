import numpy as np
import os
import argparse

def generate_random_map(width, height, obstacle_density, seed=None):
    rng = np.random.default_rng(seed)
    map_data = [["." for x in range(width)] for y in range(height)]
    total_obstacles = int(width * height * obstacle_density)

    obstacles_placed = 0
    while obstacles_placed < total_obstacles:
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        if map_data[y][x] == ".":
            map_data[y][x] = "@"
            obstacles_placed += 1

    return "\n".join("".join(row) for row in map_data)


def random_map_save(width, height, obstacle_density, seed):
    output_dir = f"data/map_files/"
    os.makedirs(output_dir, exist_ok=True)
    output_map_dir = os.path.join(
        output_dir, f"random-{height}-{width}-{int(obstacle_density*100)}"
    )
    os.makedirs(output_map_dir, exist_ok=True)

    map_data = generate_random_map(width, height, obstacle_density, seed)
    output_path = os.path.join(
        output_map_dir,
        f"random-{height}-{width}-{int(obstacle_density*100)}-{seed}.map",
    )
    with open(output_path, "w") as f:
        f.write("type octile\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write("map\n")
        f.write(map_data)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--width", "-w", type=int, default=32)
    args.add_argument("--height", "-he", type=int, default=32)
    args.add_argument("--obstacle_density", "-d", type=float, default=0.2)
    args.add_argument("--num_maps", "-n", type=int, default=20)
    args = args.parse_args()
    width = args.width
    height = args.height
    obstacle_density = args.obstacle_density
    num_maps = args.num_maps
    for seed in range(num_maps):
        random_map_save(width, height, obstacle_density, seed)
