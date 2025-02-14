import struct
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import glob
from tqdm import tqdm
from .utils import parse_coordinates

def get_action(cur_pos, next_pos):
    # 计算差值
    dx = next_pos[0] - cur_pos[0]
    dy = next_pos[1] - cur_pos[1]
    
    # 使用字典映射差值到动作
    action_map = {
        (0, 1): 1,   # 右
        (0, -1): 2,  # 左
        (-1, 0): 3,  # 上
        (1, 0): 4,   # 下
    }
    
    # 返回对应的动作，如果没有匹配则返回 0
    return action_map.get((dx, dy), 0)
    
def convert_path_to_bin(file_name):
    output_file = file_name.replace("path_files", "input_data").replace(".path", ".bin")
    if os.path.exists(output_file):
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read path file
    with open(file_name, "r") as f:
        lines = f.readlines()

    # Parse path data
    solution_line = -1
    for i, line in enumerate(lines):
        if line.startswith("solution="):
            solution_line = i
            break
            
    if solution_line == -1 or solution_line == len(lines) - 1:
        # 删除文件
        os.remove(file_name)
        return
    paths = []
    for line in lines[solution_line + 1 :]:
        # Parse coordinates for each timestep
        coords = parse_coordinates(line.split(":")[1])
        paths.append(coords)

    # Write all steps to a single binary file
    with open(output_file, "wb") as f:
        steps = len(paths)
        agent_num = len(paths[0])
        f.write(struct.pack("HH", steps, agent_num))  # Write steps and agent_num (2 bytes each)
        
        for t in range(steps):
            # Write current positions
            for agent_id in range(agent_num):
                cur_pos = paths[t][agent_id]
                f.write(struct.pack("BB", cur_pos[0], cur_pos[1]))  # 1 byte each
            
            # Write actions
            for agent_id in range(agent_num):
                cur_pos = paths[t][agent_id]
                next_pos = paths[t + 1][agent_id] if t + 1 < steps else cur_pos
                action = get_action(cur_pos, next_pos)
                f.write(struct.pack("B", action))  # 1 byte

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m tools.convert_lacam_path_to_bin <path_to_lacam_result_file_dir>"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    # 递归查找所有 .path 文件
    path_files = glob.glob(os.path.join(input_dir, "**/*.path"), recursive=True)

    if not path_files:
        print(f"No .path files found in directory: {input_dir}")
        sys.exit(1)

    print(f"Found {len(path_files)} .path files to process")

    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(convert_path_to_bin, path_files),
                total=len(path_files),
                desc="Converting files",
            )
        )
