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
    

if __name__ == '__main__':

    agent_numbers = [ 8, 16, 24, 32, 48, 64 ]
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
                    print(f"Observation shape: {len(obs)}")
                    print(f"Observation: {obs}")
                    result = lacam.solve(obs)
                    formatted_data = convert_paths(result)
                    # Save to a YAML file
                    with open(f'{folder}/map_{map_name}_agent_{agent_number}_seed_{seed}.yaml', 'w') as file:
                        yaml.dump(formatted_data, file, default_flow_style=False)
                
