import struct
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import glob
from tqdm import tqdm
import pickle
from .utils import create_distance_map, parse_coordinates, parse_file_name

# Add lock at module level
distance_map_lock = Lock()

def get_action(cur_pos, next_pos):
    pos_diff = tuple(np.array(next_pos) - np.array(cur_pos))
    if pos_diff == (0, 1):
        return 1
    elif pos_diff == (0, -1):
        return 2
    elif pos_diff == (-1, 0):
        return 3
    elif pos_diff == (1, 0):
        return 4
    else:
        return 0
    
def convert_path_to_bin(file_name):
    map_name, agent_num, path_name = parse_file_name(file_name)
    output_dir = file_name.split(".")[0].replace("path_files", "input_data")
    if os.path.exists(output_dir) and os.listdir(output_dir):
        return
    os.makedirs(output_dir, exist_ok=True)

    # Read path file
    with open(file_name, "r") as f:
        lines = f.readlines()

    # Parse path data
    solution_line = -1
    for i, line in enumerate(lines):
        if line.startswith("solution="):
            solution_line = i
            break
            
    if solution_line == -1:
        print(f"Error: solution line not found in file: {file_name}")
        return
    
    paths = [[] for _ in range(agent_num)]
    for line in lines[solution_line + 1 :]:
        # Parse coordinates for each timestep
        coords = parse_coordinates(line.split(":")[1])
        for agent_id, coord in enumerate(coords):
            paths[agent_id].append(coord)

    # For each timestep, create a binary file
    steps = len(paths[0])
    for t in range(steps):
        output_file = os.path.join(output_dir, f"{path_name}-{t}.bin")
        with open(output_file, "wb") as f:

            # Write current positions and goals for each agent
            for agent_id in range(agent_num):
                cur_pos = paths[agent_id][t]
                f.write(
                    struct.pack(
                        "BB",
                        cur_pos[0],
                        cur_pos[1],  # Current position
                    )
                )
            for agent_id in range(agent_num):
                goal = paths[agent_id][-1]
                f.write(
                    struct.pack(
                        "BB",
                        goal[0],
                        goal[1],  # Goal position
                    )
                )
            for agent_id in range(agent_num):
                action = 0
                cur_pos = paths[agent_id][t]
                next_pos = paths[agent_id][t + 1] if t + 1 < steps else cur_pos
                action = get_action(cur_pos, next_pos)
                f.write(struct.pack("B", action))
    distance_map_path = os.path.join(
        "data/distance_maps",
        f"{map_name}.pkl",
    )
    
    if not os.path.exists(distance_map_path):
        # Add lock around distance map creation and saving
        with distance_map_lock:
            # Check again in case another thread created it while we were waiting
            if not os.path.exists(distance_map_path):
                distance_map = create_distance_map(file_name)
                os.makedirs(os.path.dirname(distance_map_path), exist_ok=True)
                pickle.dump(distance_map, open(distance_map_path, "wb"))

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python convert_lacam_path_to_bin.py <path_to_lacam_result_file_dir>"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    # 递归查找所有 .path 文件
    path_files = glob.glob(os.path.join(input_dir, "**/*.path"), recursive=True)

    if not path_files:
        print(f"No .path files found in directory: {input_dir}")
        sys.exit(1)

    print(f"Found {len(path_files)} .path files to process")

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(convert_path_to_bin, path_files),
                total=len(path_files),
                desc="Converting files",
            )
        )
