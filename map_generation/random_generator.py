import numpy as np
import argparse
import os

class MapRangeSettings:
    width_min: int = 17
    width_max: int = 21
    height_min: int = 17
    height_max: int = 21
    obstacle_density_min: float = 0.1
    obstacle_density_max: float = 0.3

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)
        return {
            "width": rng.integers(self.width_min, self.width_max + 1),
            "height": rng.integers(self.height_min, self.height_max + 1),
            "obstacle_density": rng.uniform(self.obstacle_density_min, self.obstacle_density_max),
            "seed": seed
        }

    def manual_sample(self, width, height, obstacle_density, seed):
        return {
            "width": width,
            "height": height,
            "obstacle_density": obstacle_density,
            "seed": seed
        }
    

def generate_map(settings):
    rng = np.random.default_rng(settings["seed"])
    width, height, obstacle_density = settings["width"], settings["height"], settings["obstacle_density"]
    map_data = [['.' for _ in range(width)] for _ in range(height)]
    total_tiles = width * height
    total_obstacles = int(total_tiles * obstacle_density)

    obstacles_placed = 0
    while obstacles_placed < total_obstacles:
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        if map_data[y][x] == '.':
            map_data[y][x] = '@'
            obstacles_placed += 1

    return '\n'.join(''.join(row) for row in map_data)

def save_map_to_file(map_name, map_type, width, height, map_data, directory):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{map_name}.map")
    with open(file_path, "w") as file:
        file.write(f"type {map_type}\n")
        file.write(f"height {height}\n")
        file.write(f"width {width}\n")
        file.write("map\n")
        file.write(map_data)
    print(f"Map saved to {file_path}")

def generate_and_save_maps(number_of_maps, output_dir):
    seed_range = range(number_of_maps)
    settings_generator = MapRangeSettings()

    for seed in seed_range:
        settings = settings_generator.sample(seed)
        map_data = generate_map(settings)
        map_name = f"random_{seed + 1}"
        save_map_to_file(map_name, "octile", settings["width"], settings["height"], map_data, output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_maps", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="map_file")
    args = parser.parse_args()

    # single generate test
    # map = generate_map(MapRangeSettings().manual_sample(10, 20, 0.2, 0))
    # save_map_to_file("random_test", "octile", 10, 20, map, "map_file")

    # batch generate 5 maps
    generate_and_save_maps(args.number_of_maps, args.output_dir)

