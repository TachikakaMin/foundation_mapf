import numpy as np
import os
import pickle
from tqdm import tqdm
import torch
import random
NOT_FOUND_PATH = 128

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
    file_name = os.path.basename(file_name).split(".")[0]
    name_parts = file_name.split("-")
    map_name = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}-{name_parts[3]}-{name_parts[4]}"
    agent_num = int(name_parts[5])
    return map_name, agent_num, file_name


def read_map(map_name):
    name_parts = map_name.split("-")
    map_path = os.path.join(
        "data/map_files",
        f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}-{name_parts[3]}",
        f"{map_name}.map",
    )
    with open(map_path, "r") as f:
        map_lines = f.readlines()
        # 读取地图数据并去除每行末尾的换行符, 跳过前4行
        map_lines = [line.rstrip("\n") for line in map_lines[4:]]

    # 将地图字符转换为二进制数组
    map_data = np.zeros((len(map_lines), len(map_lines[0])))
    for i, line in enumerate(map_lines):
        for j, char in enumerate(line):
            if char == "@":
                map_data[i][j] = 1  # 障碍物标记为1
            else:
                map_data[i][j] = 0  # 可通行区域标记为0

    return map_data


def read_distance_map(map_name):
    map_path = os.path.join(
        "data/distance_maps",
        f"{map_name}.pkl",
    )
    return pickle.load(open(map_path, "rb"))


def create_distance_map(file_name):
    NOT_FOUND_PATH = 128
    map_name, _, _ = parse_file_name(file_name)
    map_data = read_map(map_name)
    from collections import deque

    n, m = map_data.shape
    dist_matrix = {}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    accessible_points = [
        (i, j) for j in range(m) for i in range(n) if map_data[i, j] == 0
    ]

    for start in tqdm(accessible_points, desc="Calculating distances"):

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

        dist_matrix[(i, j)] = dist

    return dist_matrix


def parse_coordinates(coord_str):
    """Parse LACAM coordinates"""
    # 移除所有空格和换行符
    coord_str = coord_str.strip()
    # 移除开头的 "(" 和结尾的 "," 和 ")"
    coord_str = coord_str.strip("(,)")
    # 分割成单独的坐标对
    coord_pairs = coord_str.split("),(")
    # 解析每个坐标对
    coords = []
    for pair in coord_pairs:
        # 交换x和y
        y, x = map(int, pair.split(","))
        coords.append((x, y))
    return coords
