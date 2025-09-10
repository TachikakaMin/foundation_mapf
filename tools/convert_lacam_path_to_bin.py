import struct
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import glob
from tqdm import tqdm
from collections import defaultdict
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
    
    # 返回对应的动作, 如果没有匹配则返回 0
    return action_map.get((dx, dy), 0)

def convert_path_to_scenario_data(file_name):
    """将单个路径文件转换为scenario数据"""
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
        return None
        
    paths = []
    for line in lines[solution_line + 1:]:
        # Parse coordinates for each timestep
        coords = parse_coordinates(line.split(":")[1])
        paths.append(coords)

    if not paths or not paths[0]:
        return None

    steps = len(paths)
    agent_num = len(paths[0])
    
    # 生成scenario数据
    scenario_data = bytearray()
    scenario_data.extend(struct.pack("HH", steps, agent_num))  # steps and agent_num (2 bytes each)
    
    for t in range(steps):
        # Write current positions
        for agent_id in range(agent_num):
            cur_pos = paths[t][agent_id]
            scenario_data.extend(struct.pack("BB", cur_pos[0], cur_pos[1]))  # 1 byte each
        
        # Write actions
        for agent_id in range(agent_num):
            cur_pos = paths[t][agent_id]
            next_pos = paths[t + 1][agent_id] if t + 1 < steps else cur_pos
            action = get_action(cur_pos, next_pos)
            scenario_data.extend(struct.pack("B", action))  # 1 byte
    
    return {
        'steps': steps,
        'agent_num': agent_num,
        'data': bytes(scenario_data),
        'file_name': os.path.basename(file_name)
    }

def group_path_files(path_files):
    """将路径文件按地图和代理数量分组"""
    groups = defaultdict(list)
    
    for file_path in path_files:
        # 解析文件名获取地图信息和代理数量
        # 例如: maze-32-32-60-1-75-0-16-1.path -> (maze-32-32-60-1-75, 16)
        basename = os.path.basename(file_path).replace('.path', '')
        parts = basename.split('-')
        
        if len(parts) >= 8:  # 确保格式正确
            # 重构地图名称 (maze-32-32-60-1-75-0)
            map_name = '-'.join(parts[:7])  # maze-32-32-60-1-75-0
            agent_num = parts[7]  # 16
            
            key = (map_name, agent_num)
            groups[key].append(file_path)
    
    return groups

def create_mbin_file(map_agent_key, path_files):
    """为同一地图和代理数量的文件创建.mbin文件"""
    map_name, agent_num = map_agent_key
    
    # 创建输出文件路径
    output_dir = os.path.join("data/input_data", map_name, f"{map_name}-{agent_num}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{map_name}-{agent_num}.mbin")
    
    if os.path.exists(output_file):
        return
    
    print(f"处理 {map_name}-{agent_num}: {len(path_files)} 个文件")
    
    # 转换所有路径文件为scenario数据
    scenarios = []
    for file_path in tqdm(path_files, desc=f"转换 {map_name}-{agent_num}"):
        scenario_data = convert_path_to_scenario_data(file_path)
        if scenario_data:
            scenarios.append(scenario_data)
    
    if not scenarios:
        print(f"警告: {map_name}-{agent_num} 没有有效的scenario数据")
        return
    
    # 写入.mbin文件
    with open(output_file, "wb") as f:
        # 文件头部 (16字节)
        num_scenarios = len(scenarios)
        f.write(struct.pack("I", num_scenarios))  # 4字节: scenario数量
        f.write(b'\x00' * 12)  # 12字节预留
        
        # 计算索引表和数据偏移
        header_size = 16
        index_table_size = num_scenarios * 272  # 每个索引272字节
        data_start_offset = header_size + index_table_size
        
        # 写入索引表 (每个scenario 272字节)
        current_offset = data_start_offset
        for scenario in scenarios:
            data_size = len(scenario['data'])
            steps = scenario['steps']
            agent_num = scenario['agent_num']
            file_name = scenario['file_name']
            
            # 索引数据结构: offset(8) + data_size(4) + steps(2) + agent_num(2) + file_name(256)
            index_data = struct.pack("Q", current_offset)  # offset (8字节)
            index_data += struct.pack("I", data_size)      # data_size (4字节)
            index_data += struct.pack("H", steps)          # steps (2字节)
            index_data += struct.pack("H", agent_num)      # agent_num (2字节)
            
            # 文件名 (256字节，不足的用0填充)
            file_name_bytes = file_name.encode('utf-8')[:255]  # 确保不超过255字节
            file_name_padded = file_name_bytes + b'\x00' * (256 - len(file_name_bytes))
            index_data += file_name_padded
            
            f.write(index_data)
            current_offset += data_size
        
        # 写入scenario数据
        for scenario in scenarios:
            f.write(scenario['data'])
    
    print(f"✅ 生成 {output_file}, 包含 {num_scenarios} 个scenarios")

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
    
    # 将文件按地图和代理数量分组
    grouped_files = group_path_files(path_files)
    print(f"分组后共有 {len(grouped_files)} 个不同的地图-代理组合")
    
    # 为每个组创建.mbin文件
    for map_agent_key, files in grouped_files.items():
        create_mbin_file(map_agent_key, files)
    
    print("✅ 所有.mbin文件生成完成")
