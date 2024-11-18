from pogema import GridConfig, pogema_v0

import yaml

from lacam.inference2 import LacamInference

def convert_paths(agent_paths):
    formatted_data = {}
    for i, agent_path in enumerate(agent_paths):
        formatted_data[f'agent{i}'] = [
            {'x': coord[0], 'y': coord[1], 't': t} 
            for t, coord in enumerate(agent_path)
        ]
    return formatted_data

def grid_to_str(self):
    grid = [['.' if cell == self.config.FREE else '#' for cell in row] for row in self.obstacles]

    # Mark agents with 'A' on the grid
    for agent_pos in self.positions_xy:
        x, y = agent_pos
        grid[x][y] = 'A'

    # Mark targets with 'T' on the grid
    for target_pos in self.finishes_xy:
        x, y = target_pos
        # Avoid overwriting an agent position with a target symbol
        if grid[x][y] != 'A':
            grid[x][y] = 'T'

    # Convert the grid to a string
    return '\n'.join(''.join(row) for row in grid)


if __name__ == '__main__':

    agent_numbers = [ 8 ]
    folder_names = [
        'maze_maps',
    ]
    lacam = LacamInference()

    for folder in folder_names:
        maps_path = folder + "/test_map.yaml"
        print(f"Loading maps from {maps_path}")
        with open(maps_path, 'r') as f:
            maps = yaml.safe_load(f)
        for map_name, map_value in maps.items():
            for agent_number in agent_numbers:
                for seed in range(1):
                    print(f"Running map {map_name} with {agent_number} agents and seed {seed}")
                    config = GridConfig(map=map_value, num_agents=agent_number, observation_type="MAPF", seed=seed)
                    env = pogema_v0(config)
                    obs, _ = env.reset()
                    print(grid_to_str(env.grid))
                    result = lacam.solve(obs)
                    formatted_data = convert_paths(result)
                    # Save to a YAML file
                    with open(f'{folder}/map_{map_name}_agent_{agent_number}_seed_{seed}.yaml', 'w') as file:
                        yaml.dump(formatted_data, file, default_flow_style=False)
                
