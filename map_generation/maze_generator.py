import numpy as np

def maps_dict_to_yaml(filename, maps):
    import yaml
    with open(filename, 'w') as file:
        yaml.add_representer(str,
                             lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|'))
        yaml.dump(maps, file)

class MazeRangeSettings:
    width_min: int = 5
    width_max: int = 9

    height_min: int = 5
    height_max: int = 9

    obstacle_density_min: float = 0.0
    obstacle_density_max: float = 1.0

    wall_components_min: int = 1
    wall_components_max: int = 8

    go_straight_min: float = 0.75
    go_straight_max: float = 0.85

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)

        # Generate a sample for each attribute
        width = rng.integers(self.width_min, self.width_max + 1)
        height = rng.integers(self.height_min, self.height_max + 1)
        obstacle_density = rng.uniform(self.obstacle_density_min, self.obstacle_density_max)
        wall_components = rng.integers(self.wall_components_min, self.wall_components_max + 1)
        go_straight = rng.uniform(self.go_straight_min, self.go_straight_max)

        # Return a dictionary with the sampled values
        return {
            "width": width,
            "height": height,
            "obstacle_density": obstacle_density,
            "wall_components": wall_components,
            "go_straight": go_straight,
            "seed": seed,
        }

    def manual_sample(self, width, height, obstacle_density, wall_components, go_straight, seed):
        return {
            "width": width,
            "height": height,
            "obstacle_density": obstacle_density,
            "wall_components": wall_components,
            "go_straight": go_straight,
            "seed": seed,
        }


# Adapted from https://github.com/marmotlab/PRIMAL2/blob/main/Map_Generator.py
class MazeGenerator:

    @classmethod
    def array_to_string(cls, array_maze):
        result = []
        for line in array_maze:
            result.append("".join(['#' if x == 1 else '.' for x in line]))

        map_str = '\n'.join([''.join(row) for row in result])
        return map_str

    @staticmethod
    def select_random_neighbor(x, y, maze_grid, maze_shape, rng, last_direction, go_straight):
        neighbor_coords = []
        probabilities = []
        if x > 1:
            neighbor_coords.append((y, x - 2))
            probabilities.append(
                go_straight if (y, x - 2) == (y + last_direction[0], x + last_direction[1]) else (1 - go_straight))
        if x < maze_shape[1] - 2:
            neighbor_coords.append((y, x + 2))
            probabilities.append(
                go_straight if (y, x + 2) == (y + last_direction[0], x + last_direction[1]) else (1 - go_straight))
        if y > 1:
            neighbor_coords.append((y - 2, x))
            probabilities.append(
                go_straight if (y - 2, x) == (y + last_direction[0], x + last_direction[1]) else (1 - go_straight))
        if y < maze_shape[0] - 2:
            neighbor_coords.append((y + 2, x))
            probabilities.append(
                go_straight if (y + 2, x) == (y + last_direction[0], x + last_direction[1]) else (1 - go_straight))

        if not neighbor_coords:
            return None, None, last_direction

        if all(prob == go_straight for prob in probabilities):  # Adjust probabilities if all are biased
            probabilities = [1 / len(probabilities)] * len(probabilities)
        else:
            total = sum(probabilities)
            probabilities = [prob / total for prob in probabilities]

        chosen_index = rng.choice(range(len(neighbor_coords)), p=probabilities)
        next_y, next_x = neighbor_coords[chosen_index]
        new_direction = (next_y - y, next_x - x)
        return next_x, next_y, new_direction

    @classmethod
    def generate_maze(cls, width, height, obstacle_density, wall_components, go_straight, seed=None):

        rng = np.random.default_rng(seed)
        assert width > 0 and height > 0, "Width and height must be positive integers"
        maze_shape = ((height // 2) * 2 + 3, (width // 2) * 2 + 3)
        density = int(
            maze_shape[0] * maze_shape[1] * obstacle_density // wall_components) if wall_components != 0 else 0

        maze_grid = np.zeros(maze_shape, dtype='int')
        maze_grid[0, :] = maze_grid[-1, :] = 1
        maze_grid[:, 0] = maze_grid[:, -1] = 1

        for i in range(density):
            x = rng.integers(0, maze_shape[1] // 2) * 2
            y = rng.integers(0, maze_shape[0] // 2) * 2
            maze_grid[y, x] = 1
            last_direction = (0, 0)  # Initial direction is null
            for j in range(wall_components):
                next_x, next_y, last_direction = MazeGenerator.select_random_neighbor(
                    x, y, maze_grid, maze_shape, rng, last_direction, go_straight
                )
                if next_x is not None and maze_grid[next_y, next_x] == 0:
                    maze_grid[next_y, next_x] = 1
                    maze_grid[next_y + (y - next_y) // 2, next_x + (x - next_x) // 2] = 1
                    x, y = next_x, next_y

        return cls.array_to_string(maze_grid[1:-1, 1:-1])

def generate_and_save_maps(name_prefix, number_of_maps):
    maps = {}
    seed_range = range(number_of_maps)
    max_digits = len(str(number_of_maps))
    for seed in seed_range:
        settings = MazeRangeSettings().sample(seed)
        maze = MazeGenerator.generate_maze(**settings)
        map_name = f"seed-{str(seed).zfill(max_digits)}"
        maps[map_name] = maze
    maps_dict_to_yaml(f'{name_prefix}.yaml', maps)


if __name__ == "__main__":
    import os
    os.makedirs("maze_maps", exist_ok=True)

    # single generate
    map = {"map_0": MazeGenerator.generate_maze(**MazeRangeSettings().manual_sample(10, 20, 0.2, 3, 0.8, 0))}
    maps_dict_to_yaml(f'maze_maps/test_maze_map.yaml', map)

    # batch generate 128 maps
    generate_and_save_maps("maze_maps/maps", 128)