import os
from tqdm import tqdm
from pogema import GridConfig, pogema_v0
from lacam.inference import LacamInference

import yaml
import argparse
from multiprocessing import Pool

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

# Add this new function to handle individual seed processing
def process_seed(args):
    map_name, map_content, agent_number, seed = args
    yaml_filename = f'data/{map_name}/agent_{agent_number}_seed_{seed}.yaml'
    
    # Skip if file already exists
    if os.path.exists(yaml_filename):
        return
    
    config = GridConfig(map=map_content, num_agents=agent_number, observation_type="MAPF", seed=seed)
    env = pogema_v0(config)
    obs, _ = env.reset()
    lacam = LacamInference()
    result = lacam.solve(obs)
    if result is None:
        print(f"Lacam failed to find path for {agent_number} agents")
        return
    formatted_data = convert_paths(result, map_name)
    with open(yaml_filename, 'w') as file:
        yaml.dump(formatted_data, file, default_flow_style=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_range', type=int, default=4, help='Number of seeds to generate start and target positions')
    parser.add_argument('--agent_numbers', type=int, default=64)
    parser.add_argument('--map_file', type=str, default='map_file/random-32-32-20.map', help='Map file')
    args = parser.parse_args()
    map_name = args.map_file.split('/')[-1].split('.')[0]
    os.makedirs('data/' + map_name, exist_ok=True)
    map_content = read_map_file(args.map_file)
            
    agent_numbers = [i for i in range(args.agent_numbers, 2, -2)]
    for agent_number in tqdm(agent_numbers, desc='Processing agent counts'):
        # Create arguments for parallel processing
        args_list = [(map_name, map_content, agent_number, seed) 
                    for seed in range(args.seed_range)]
        
        # Use multiprocessing to process seeds in parallel
        with Pool() as pool:
            pool.map(process_seed, args_list)
        