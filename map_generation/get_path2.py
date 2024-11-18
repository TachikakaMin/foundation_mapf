from pogema import GridConfig, pogema_v0

import yaml



if __name__ == '__main__':

    agent_numbers = [ 8, 16, 24, 32, 48, 64 ]
    folder_names = [
        'random_maps',
    ]

    for folder in folder_names:
        maps_path = folder + "/maps.yaml"
        print(f"Loading maps from {maps_path}")
        with open(maps_path, 'r') as f:
            maps = yaml.safe_load(f)
        for map_name, map_value in maps.items():
            for agent_number in agent_numbers:
                print(f"Running map {map_name} with {agent_number} agents")
                config = GridConfig(map={map_name}, num_agents=agent_number)
                env = pogema_v0(config)
                env._initialize_grid()
                print(env.grid.obstacles)
