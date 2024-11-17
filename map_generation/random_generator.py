import numpy as np


def maps_dict_to_yaml(filename, maps):
    import yaml
    with open(filename, 'w') as file:
        yaml.add_representer(str,
                             lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|'))
        yaml.dump(maps, file)


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
            map_data[y][x] = '#'
            obstacles_placed += 1

    return '\n'.join(''.join(row) for row in map_data)


def generate_and_save_maps(name_prefix, number_of_maps):
    maps = {}
    seed_range = range(number_of_maps)
    max_digits = len(str(number_of_maps))
    settings_generator = MapRangeSettings()

    for seed in seed_range:
        settings = settings_generator.sample(seed)
        map_data = generate_map(settings)
        map_name = f"seed-{str(seed).zfill(max_digits)}"
        maps[map_name] = map_data

    maps_dict_to_yaml(f'{name_prefix}.yaml', maps)


if __name__ == "__main__":
    import os
    os.makedirs("random_maps", exist_ok=True)
    # single generate
    map = {"map_0": generate_map(MapRangeSettings().manual_sample(10, 20, 0.2, 0))}
    maps_dict_to_yaml(f'random_maps/test_random_map.yaml', map)

    # batch generate 128 maps
    generate_and_save_maps("random_maps/maps", 128)

