import os
from pogema import GridConfig, pogema_v0
from lacam.inference import LacamInference

import yaml
import argparse

def convert_paths(agent_paths, map_file_name):
    # Convert agent paths into the required format
    formatted_data = {}
    for i, agent_path in enumerate(agent_paths):
        formatted_data[f'agent{i}'] = [
            {'x': coord[0], 'y': coord[1], 't': t} 
            for t, coord in enumerate(agent_path)
        ]
    
    # Add the statistics section
    statistics = {
        'cost': 1291,
        'newnode_runtime': 0.000230253,
        'focal_score_time': 0.0906343,
        'firstconflict_runtime': 0.289567,
        'runtime': 0.96764,
        'lowlevel_search_time': 0.664929,
        'total_lowlevel_node': 49,
        'lowLevelExpanded': 45255,
        'numTaskAssignments': 0,
        'map': map_file_name + ".map"  # Use the actual map file name
    }
    
    # Combine statistics and schedule
    output_data = {
        'statistics': statistics,
        'schedule': formatted_data
    }
    
    return output_data

def grid_to_str(grid):
    string = [['.' if cell == grid.config.FREE else '#' for cell in row] for row in grid.obstacles]

    # Mark agents with 'A' on the grid
    for agent_pos in grid.positions_xy:
        x, y = agent_pos
        string[x][y] = 'A'

    # Mark targets with 'T' on the grid
    for target_pos in grid.finishes_xy:
        x, y = target_pos
        # Avoid overwriting an agent position with a target symbol
        if string[x][y] != 'A':
            string[x][y] = 'T'

    # Convert the grid to a string
    return '\n'.join(''.join(row) for row in string)

def read_map_file(file_path):
    """Reads a .map file and extracts the map portion as a string."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # The map starts from the 5th line (index 4)
    map_content = ''.join(lines[4:])  # Join all map lines into a single string
    return map_content.strip()  # Remove any trailing whitespace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_range', type=int, default=4, help='Number of seeds to generate start and target positions')
    parser.add_argument('--agent_numbers', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--folder', type=str, default='map_file', help='Folder containing .map files')
    args = parser.parse_args()

    seed_range = args.seed_range
    agent_numbers = args.agent_numbers
    folder_name = args.folder
    lacam = LacamInference()
    os.makedirs('data', exist_ok=True)
    # Iterate through all .map files in the specified folder
    for file_name in os.listdir(folder_name):
        if file_name.endswith("16-16.map"):
            map_path = os.path.join(folder_name, file_name)
            map_name = os.path.splitext(file_name)[0]  # Extract base name without extension

            map_content = read_map_file(map_path)
            # For each .map file, process with different agent numbers and seeds
            for agent_number in agent_numbers:
                for seed in range(seed_range):
                    print(f"Running map {map_name} with {agent_number} agents and seed {seed}")
                    config = GridConfig(map=map_content, num_agents=agent_number, observation_type="MAPF", seed=seed)
                    env = pogema_v0(config)
                    obs, _ = env.reset()
                    print(grid_to_str(env.grid))
                    result = lacam.solve(obs)

                    formatted_data = convert_paths(result, map_name)
                    yaml_filename = f'data/{map_name}_agent_{agent_number}_seed_{seed}.yaml'
                    with open(yaml_filename, 'w') as file:
                        yaml.dump(formatted_data, file, default_flow_style=False)
