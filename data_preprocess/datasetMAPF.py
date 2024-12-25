import os
import traceback
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq

def calculate_minimum_distance(current_agent_location, goal_agent_location, map_info):
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def a_star(start, goal, grid):
        if tuple(start) == tuple(goal):
            return 0

        rows, cols = grid.shape
        start = tuple(start)
        goal = tuple(goal)

        open_set = [(manhattan_distance(start, goal), 0, start)]
        g_score = {start: 0}
        visited = set()
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current == goal:
                return current_g

            if current in visited:
                continue
            visited.add(current)

            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)

                if 0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols and grid[next_pos[0], next_pos[1]] == 0:
                    tentative_g = current_g + 1

                    if next_pos not in g_score or tentative_g < g_score[next_pos]:
                        g_score[next_pos] = tentative_g
                        f_score = tentative_g + manhattan_distance(next_pos, goal)
                        heapq.heappush(open_set, (f_score, tentative_g, next_pos))

        return 10000  # No path found

    return a_star(current_agent_location, goal_agent_location, map_info)

def create_distance_map(map_data):
    # Get dimensions and find all accessible points
    rows, cols = map_data.shape
    accessible_points = [(i, j) for i in range(rows) for j in range(cols) if map_data[i, j] == 0]
    
    # Create distance dictionary
    distance_map = {}
    
    # Calculate distances between all pairs of accessible points
    for start in tqdm(accessible_points, desc="Calculating distances"):
        for goal in accessible_points:
            distance = calculate_minimum_distance(start, goal, map_data)
            distance_map[(start, goal)] = distance
    
    return distance_map

class MAPFDataset(Dataset):
    def __init__(self, h5_files, feature_dim):
        self.feature_dim = feature_dim
        self.h5_files = h5_files
        self.train_data_len = []
        self.train_data_map_name = []
        self.train_data_agent_locations = []
        self.all_map_data = {}
        self.all_distance_maps = {}
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.parallel_load_data(self.h5_files)
        
        self.train_data_len = np.array(self.train_data_len, dtype=int)
        self.train_cumsum_len = np.cumsum(self.train_data_len)
        for map_name in self.all_map_data.keys():
            distance_cache_file = os.path.join(self.cache_dir, f"{os.path.basename(map_name)}_distance.pkl")
        
            if not os.path.exists(distance_cache_file):
                map_data = self.all_map_data[map_name].numpy()
                distance_map = create_distance_map(map_data)
                with open(distance_cache_file, 'wb') as f:
                    pickle.dump(distance_map, f)
            else:
                with open(distance_cache_file, 'rb') as f:
                    distance_map = pickle.load(f)
            self.all_distance_maps[map_name] = distance_map

    def parallel_load_data(self, h5_files):
        self.train_data, self.action_data = [], []
        with ThreadPoolExecutor(max_workers=128) as executor:
            futures = {executor.submit(self.process_h5_file, h5_file): h5_file for h5_file in h5_files}
            for future in tqdm(as_completed(futures), total=len(h5_files)):
                try:
                    map_name, agent_locations, h5_file = future.result()
                    if agent_locations.shape[0] == 0:
                        continue
                    self.train_data_len.append(agent_locations.shape[1] - 1)
                    self.train_data_map_name.append(map_name)
                    self.train_data_agent_locations.append(agent_locations)
                except Exception as e:
                    print(traceback.format_exc())
                    invalid_file = futures[future]
                    print(f"Invalid file detected and removed: {invalid_file}")
                    os.remove(invalid_file)

    def process_h5_file(self, h5_file):
        cache_file = os.path.join(self.cache_dir, f"{os.path.basename(h5_file)}.npy")
        
        with h5py.File(h5_file, "r") as f:
            map_name = f['/statistics'].attrs['map']
            if not (map_name in self.all_map_data.keys()):
                self.all_map_data[map_name] = torch.FloatTensor(self.read_map(map_name))
                
        if os.path.exists(cache_file):
            try:
                cached_data = np.load(cache_file)
                return map_name, cached_data, h5_file
            except Exception as e:
                print(f"Cache load failed for {h5_file}, processing file: {e}")
                
        with h5py.File(h5_file, "r") as f:
            agent_locations = self.preprocess_h5_data(f)
            try:
                np.save(cache_file, agent_locations)
            except Exception as e:
                print(f"Failed to save cache for {h5_file}: {e}")
                
        return map_name, agent_locations, h5_file

    def get_distance(self, start, goal, map_name):
        # Convert start and goal to tuples
        start = tuple(start)
        goal = tuple(goal)
        
        # Directly look up in the dictionary
        return self.all_distance_maps[map_name].get((start, goal), 10000)
    
    def read_map(self, map_name):
        map_path = f"map_file/{map_name}"
        map_data = []
        with open(map_path, "r") as f:
            data = f.readlines()
            for l in data[4:]:
                tmp = l.strip()
                tmp = tmp.replace(".", "0").replace("@", "1").replace("#", "1")
                map_data.append(tmp)
        map_data = np.array([list(line) for line in map_data], dtype=int)
        
        return map_data
    
    def preprocess_h5_data(self, h5_file):
        agent_names = list(h5_file['/schedule'].keys())
        agent_num = len(agent_names)
        agent_locations = []
        max_length = 0

        for agent_name in agent_names:
            agent_path = h5_file[f'/schedule/{agent_name}/trajectory'][:]
            agent_locs = agent_path.tolist()
            max_length = max(max_length, len(agent_locs))
            agent_locations.append(agent_locs)

        for i in range(agent_num):
            sublist = agent_locations[i]
            if len(sublist) < max_length:
                last_x, last_y, last_t = sublist[-1]
                additional_steps = [[last_x, last_y, last_t + j + 1] for j in range(max_length - len(sublist))]
                agent_locations[i] += additional_steps

        agent_locations = np.array(agent_locations, dtype=int)
        return agent_locations
    
    def new_generate_train_data_one(self, agent_locations, map_info, idx, map_name):
        m, n = map_info.shape[0], map_info.shape[1]
        feature = torch.zeros((self.feature_dim, m, n))
        feature[0] = map_info
        current_agent_locations = agent_locations[:, idx, :2]
        goal_agent_locations = agent_locations[:, -1, :2]
        last_agent_locations_1 = agent_locations[:, idx-1, :2] if idx > 0 else current_agent_locations
        last_agent_locations_2 = agent_locations[:, idx-2, :2] if idx > 1 else last_agent_locations_1
        last_agent_locations_3 = agent_locations[:, idx-3, :2] if idx > 2 else last_agent_locations_2
        last_agent_locations_4 = agent_locations[:, idx-4, :2] if idx > 3 else last_agent_locations_3
        # last_agent_locations_5 = agent_locations[:, idx-5, :2] if idx > 4 else last_agent_locations_4

        for i in range(current_agent_locations.shape[0]):
            feature[1, current_agent_locations[i, 0], current_agent_locations[i, 1]] = i+1
            feature[2, goal_agent_locations[i, 0], goal_agent_locations[i, 1]] = i+1
            dx = self.get_distance(
                (current_agent_locations[i, 0]+1, current_agent_locations[i, 1]),
                goal_agent_locations[i],
                map_name
            ) - self.get_distance(
                (current_agent_locations[i, 0]-1, current_agent_locations[i, 1]),
                goal_agent_locations[i],
                map_name
            )
            dy = self.get_distance(
                (current_agent_locations[i, 0], current_agent_locations[i, 1]+1),
                goal_agent_locations[i],
                map_name
            ) - self.get_distance(
                (current_agent_locations[i, 0], current_agent_locations[i, 1]-1),
                goal_agent_locations[i],
                map_name
            )
            # normalize
            norm = (dx ** 2 + dy ** 2) ** 0.5
            dx = dx / norm if norm > 0 else 0
            dy = dy / norm if norm > 0 else 0
            feature[3, current_agent_locations[i, 0], current_agent_locations[i, 1]] = dx
            feature[4, current_agent_locations[i, 0], current_agent_locations[i, 1]] = dy
            feature[5, last_agent_locations_1[i, 0], last_agent_locations_1[i, 1]] = i+1
            feature[6, last_agent_locations_2[i, 0], last_agent_locations_2[i, 1]] = i+1
            feature[7, last_agent_locations_3[i, 0], last_agent_locations_3[i, 1]] = i+1
            feature[8, last_agent_locations_4[i, 0], last_agent_locations_4[i, 1]] = i+1
            # feature[9, last_agent_locations_5[i, 0], last_agent_locations_5[i, 1]] = i+1

        action_info = torch.zeros((m, n), dtype=torch.long)
        next_agent_locations = agent_locations[:, idx+1, :2]
        next_distance_to_goal = next_agent_locations - current_agent_locations

        for i in range(current_agent_locations.shape[0]):
            location_diff = tuple(next_distance_to_goal[i])   
            if location_diff == (0, 0): 
                action_info[current_agent_locations[i, 0], current_agent_locations[i, 1]] = 0
            elif location_diff == (0, 1):
                action_info[current_agent_locations[i, 0], current_agent_locations[i, 1]] = 1
            elif location_diff == (0, -1):
                action_info[current_agent_locations[i, 0], current_agent_locations[i, 1]] = 2
            elif location_diff == (-1, 0):
                action_info[current_agent_locations[i, 0], current_agent_locations[i, 1]] = 3
            elif location_diff == (1, 0):
                action_info[current_agent_locations[i, 0], current_agent_locations[i, 1]] = 4

        mask = torch.zeros((m, n))
        for i in range(current_agent_locations.shape[0]):
            mask[current_agent_locations[i, 0], current_agent_locations[i, 1]] = 1
        return feature, action_info, mask
        
    def __len__(self):
        return self.train_cumsum_len[-1]

    def __getitem__(self, idx):
        data_index = np.searchsorted(self.train_cumsum_len, idx, side='right')
        if data_index != 0:
            diff = idx - self.train_cumsum_len[data_index - 1]
        else:
            diff = idx  
        map_name = self.train_data_map_name[data_index]
        map_data = self.all_map_data[map_name]
        agent_locations = self.train_data_agent_locations[data_index]
        try:
            feature, action_info, mask = self.new_generate_train_data_one(agent_locations, map_data, diff, map_name)
        except Exception as e:
            print(e, map_name, data_index, diff)
            raise e
        ret_data = {"feature": feature.float(), "action": action_info, "mask": mask}
        return ret_data


    
    
    

        

    
    
