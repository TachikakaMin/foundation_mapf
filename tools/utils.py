import numpy as np
import os
import pickle
from tqdm import tqdm
import torch
import random
from collections import deque
from multiprocessing import Pool
import re
NOT_FOUND_PATH = 2048

def get_distance(distance_map, agent_location, goal_location):
    agent_location = (int(agent_location[0]), int(agent_location[1]))
    goal_location = (int(goal_location[0]), int(goal_location[1]))
    if agent_location not in distance_map:
        return NOT_FOUND_PATH
    return distance_map[agent_location][goal_location[0]][goal_location[1]]

def construct_input_feature(
    map_data,
    agent_locations,
    goal_locations,
    distance_map,
    feature_dim,
    feature_type,
):
    height, width = map_data.shape
    agent_num = agent_locations.shape[0]
    device = agent_locations.device
    input_features = torch.zeros(
        (feature_dim, height, width), dtype=torch.float32, device=device
    )
    if isinstance(map_data, torch.Tensor):
        input_features[0] = map_data.clone().detach().to(dtype=torch.float32, device=device)
    else:
        input_features[0] = torch.tensor(map_data, dtype=torch.float32, device=device)
    # 使用向量化操作设置agent位置和目标位置
    agent_indices = torch.arange(1, agent_num + 1, dtype=torch.float32, device=device)
    input_features[1, agent_locations[:, 0], agent_locations[:, 1]] = agent_indices
    input_features[2, goal_locations[:, 0], goal_locations[:, 1]] = agent_indices
    if feature_dim >= 4:
        # 批量计算距离
        distances = torch.zeros(agent_num, dtype=torch.float32, device=device)
        for i in range(agent_num):
            distances[i] = get_distance(distance_map, agent_locations[i], goal_locations[i])
        input_features[3, agent_locations[:, 0], agent_locations[:, 1]] = distances

    if feature_dim == 6:
        if feature_type == "gradient":
            dx = torch.zeros(agent_num, dtype=torch.float32, device=device)
            dy = torch.zeros(agent_num, dtype=torch.float32, device=device)
            for i in range(agent_num):
                left_distances = get_distance(distance_map, (agent_locations[i, 0] - 1, agent_locations[i, 1]), goal_locations[i])-distances[i]
                right_distances = get_distance(distance_map, (agent_locations[i, 0] + 1, agent_locations[i, 1]), goal_locations[i])-distances[i]
                up_distances = get_distance(distance_map, (agent_locations[i, 0], agent_locations[i, 1] + 1), goal_locations[i])-distances[i]
                down_distances = get_distance(distance_map, (agent_locations[i, 0], agent_locations[i, 1] - 1), goal_locations[i])-distances[i]
                if left_distances > 0 and right_distances > 0:
                    dx[i] = 0
                elif left_distances >= 0 and right_distances < 0:
                    dx[i] = 1
                elif left_distances < 0 and right_distances >= 0:
                    dx[i] = -1
                elif left_distances < 0 and right_distances < 0:
                    dx[i] = random.choice([-1, 1])
                elif left_distances == 0 and right_distances == 0:
                    dx[i] = random.choice([-1, 0, 1])
                elif left_distances == 0 and right_distances > 0:
                    dx[i] = random.choice([0, -1])
                elif left_distances > 0 and right_distances == 0:
                    dx[i] = random.choice([0, 1])
                else:
                    dx[i] = random.choice([-1, 1])
                if down_distances > 0 and up_distances > 0:
                    dy[i] = 0
                elif down_distances >= 0 and up_distances < 0:
                    dy[i] = 1
                elif down_distances < 0 and up_distances >= 0:
                    dy[i] = -1
                elif down_distances < 0 and up_distances < 0:
                    dy[i] = random.choice([-1, 1])
                elif down_distances == 0 and up_distances == 0:
                    dy[i] = random.choice([-1, 0, 1])
                elif down_distances == 0 and up_distances > 0:
                    dy[i] = random.choice([-1, 0])
                elif down_distances > 0 and up_distances == 0:
                    dy[i] = random.choice([0, 1])
                else:
                    dy[i] = random.choice([-1, 1])
            input_features[4, agent_locations[:, 0], agent_locations[:, 1]] = dx
            input_features[5, agent_locations[:, 0], agent_locations[:, 1]] = dy

        else:
            input_features[4, agent_locations[:, 0], agent_locations[:, 1]] = (
                goal_locations[:, 0] - agent_locations[:, 0]
            ).float()
            input_features[5, agent_locations[:, 0], agent_locations[:, 1]] = (
                goal_locations[:, 1] - agent_locations[:, 1]
            ).float()
    return input_features


def parse_file_name(file_name):
    dir_prefix = file_name.split("/")[0]
    path_name = os.path.basename(file_name).split(".")[0]
    if "mapf_gpt" in file_name:
        dir_prefix_2 = file_name.split("/")[2]
        name_parts = path_name.split("-")
        if ".path" in file_name:
            map_name = "-".join(name_parts[:-2])
        else:
            map_name = "-".join(name_parts[:-3])
        map_file_path = os.path.join(dir_prefix, "map_files", dir_prefix_2, f"{map_name}.map")
    elif "data_benchmark" in file_name:
        if "even" in path_name:
            map_name = path_name.split("-even")[0]
            map_file_path = os.path.join(dir_prefix, "map_files", f"{map_name}.map")
        elif "random" in path_name:
            map_name = path_name.split("-random")[0]
            map_file_path = os.path.join(dir_prefix, "map_files", f"{map_name}.map")
    else:
        name_parts = path_name.split("-")
        map_name = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}-{name_parts[3]}-{name_parts[4]}-{name_parts[5]}-{name_parts[6]}"
        map_folder = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}-{name_parts[3]}-{name_parts[4]}-{name_parts[5]}"
        map_file_path = os.path.join(dir_prefix, "map_files", map_folder, f"{map_name}.map")
    return map_file_path, path_name


def read_map(map_file_path):
    with open(map_file_path, "r") as f:
        map_lines = f.readlines()
        height = int(map_lines[1].split(" ")[1])
        width = int(map_lines[2].split(" ")[1])
        # 读取地图数据并去除每行末尾的换行符, 跳过前4行
        map_lines = [line.rstrip("\n") for line in map_lines[4:]]

    # 将地图字符转换为二进制数组
    map_data = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if map_lines[i][j] == "@" or map_lines[i][j] == "T":
                map_data[i][j] = 1  # 障碍物标记为1
            else:
                map_data[i][j] = 0  # 可通行区域标记为0

    return map_data


def read_distance_map(map_file_path):
    file_path = map_file_path.replace("map_files", "distance_maps").replace(".map", ".pkl")
    return pickle.load(open(file_path, "rb"))

def calculate_single_point_distances(args):
    start, map_data, n, m = args
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    i, j = start
    dist = np.full((n, m), fill_value=NOT_FOUND_PATH, dtype=np.int32)
    dist[i][j] = 0

    queue = deque()
    queue.append((i, j))

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m:
                if map_data[nx][ny] == 0 and dist[nx][ny] == NOT_FOUND_PATH:
                    dist[nx][ny] = dist[x][y] + 1
                    queue.append((nx, ny))
    
    return (start, dist)


def create_distance_map(map_data):

    n, m = map_data.shape
    accessible_points = [
        (i, j) for j in range(m) for i in range(n) if map_data[i, j] == 0
    ]
    
    # 准备并行处理的参数
    args = [(start, map_data, n, m) for start in accessible_points]
    
    # 使用CPU核心数作为进程数
    # num_processes = multiprocessing.cpu_count()
    
    # 创建进程池并执行并行计算
    with Pool(processes=1) as pool:
        results = list(tqdm(
            pool.imap(calculate_single_point_distances, args),
            total=len(accessible_points),
            desc="Calculating distances"
        ))
    
    # 将结果转换为字典
    dist_matrix = dict(results)
    
    return dist_matrix


def parse_coordinates(coord_str):
    """Parse LACAM coordinates"""
    # 使用正则表达式提取所有坐标对
    coord_pairs = re.findall(r"\((\d+),(\d+)\)", coord_str)
    
    # 解析每个坐标对，并交换 x 和 y
    coords = [(int(y), int(x)) for x, y in coord_pairs]
    
    return coords
