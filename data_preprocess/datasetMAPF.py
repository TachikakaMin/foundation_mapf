import os
import traceback
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import jit

@jit(nopython=True)
def calculate_minimum_distance(current_agent_location, goal_agent_location, map_info):
    # Early return for same positions
    if current_agent_location[0] == goal_agent_location[0] and current_agent_location[1] == goal_agent_location[1]:
        return 0
    
    # Quick boundary check
    rows, cols = map_info.shape
    if (not 0 <= current_agent_location[0] < rows or 
        not 0 <= current_agent_location[1] < cols or
        not 0 <= goal_agent_location[0] < rows or 
        not 0 <= goal_agent_location[1] < cols or
        map_info[current_agent_location[0], current_agent_location[1]] == 1 or 
        map_info[goal_agent_location[0], goal_agent_location[1]] == 1):
        return 128

    # Initialize data structures
    visited = np.zeros((rows, cols), dtype=np.bool_)
    g_scores = np.full((rows, cols), np.inf)
    g_scores[current_agent_location[0], current_agent_location[1]] = 0
    
    # Priority queue simulation using arrays
    queue_size = rows * cols
    queue_len = 0
    queue_f = np.full(queue_size, np.inf)
    queue_g = np.full(queue_size, np.inf)
    queue_x = np.zeros(queue_size, dtype=np.int32)
    queue_y = np.zeros(queue_size, dtype=np.int32)
    
    # Add start node
    f_start = abs(current_agent_location[0] - goal_agent_location[0]) + abs(current_agent_location[1] - goal_agent_location[1])
    queue_f[0] = f_start
    queue_g[0] = 0
    queue_x[0] = current_agent_location[0]
    queue_y[0] = current_agent_location[1]
    queue_len = 1

    directions = np.array([(0, 1), (0, -1), (-1, 0), (1, 0)])

    while queue_len > 0:
        # Find minimum f_score in queue
        min_idx = 0
        min_f = queue_f[0]
        for i in range(1, queue_len):
            if queue_f[i] < min_f:
                min_f = queue_f[i]
                min_idx = i
        
        current_x = queue_x[min_idx]
        current_y = queue_y[min_idx]
        current_g = queue_g[min_idx]
        
        # Remove from queue by swapping with last element and reducing length
        queue_len -= 1
        queue_f[min_idx] = queue_f[queue_len]
        queue_g[min_idx] = queue_g[queue_len]
        queue_x[min_idx] = queue_x[queue_len]
        queue_y[min_idx] = queue_y[queue_len]
        
        if current_x == goal_agent_location[0] and current_y == goal_agent_location[1]:
            return int(current_g)
            
        if visited[current_x, current_y]:
            continue
            
        visited[current_x, current_y] = True
        
        for dx, dy in directions:
            next_x = current_x + dx
            next_y = current_y + dy
            
            if (0 <= next_x < rows and 0 <= next_y < cols and 
                not visited[next_x, next_y] and map_info[next_x, next_y] == 0):
                
                tentative_g = current_g + 1
                
                if tentative_g < g_scores[next_x, next_y]:
                    g_scores[next_x, next_y] = tentative_g
                    f_score = tentative_g + abs(next_x - goal_agent_location[0]) + abs(next_y - goal_agent_location[1])
                    
                    if queue_len < queue_size:
                        queue_f[queue_len] = f_score
                        queue_g[queue_len] = tentative_g
                        queue_x[queue_len] = next_x
                        queue_y[queue_len] = next_y
                        queue_len += 1

    return 128  # No path found

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
        return self.all_distance_maps[map_name].get((start, goal), 128)
    
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
        # last_agent_locations_2 = agent_locations[:, idx-2, :2] if idx > 1 else last_agent_locations_1
        # last_agent_locations_3 = agent_locations[:, idx-3, :2] if idx > 2 else last_agent_locations_2
        # last_agent_locations_4 = agent_locations[:, idx-4, :2] if idx > 3 else last_agent_locations_3
        # last_agent_locations_5 = agent_locations[:, idx-5, :2] if idx > 4 else last_agent_locations_4

        for i in range(current_agent_locations.shape[0]):
            feature[1, current_agent_locations[i, 0], current_agent_locations[i, 1]] = i+1
            feature[2, goal_agent_locations[i, 0], goal_agent_locations[i, 1]] = i+1
            feature[3, last_agent_locations_1[i, 0], last_agent_locations_1[i, 1]] = i+1

            # x = goal_agent_locations[i, 0] - current_agent_locations[i, 0]
            # y = goal_agent_locations[i, 1] - current_agent_locations[i, 1]
            # feature[3, current_agent_locations[i, 0], current_agent_locations[i, 1]] = x
            # feature[4, current_agent_locations[i, 0], current_agent_locations[i, 1]] = y
            # feature[7, last_agent_locations_2[i, 0], last_agent_locations_2[i, 1]] = i+1
            # feature[8, last_agent_locations_3[i, 0], last_agent_locations_3[i, 1]] = i+1
            # feature[9, last_agent_locations_4[i, 0], last_agent_locations_4[i, 1]] = i+1
            # feature[10, last_agent_locations_5[i, 0], last_agent_locations_5[i, 1]] = i+1

        for i in range(current_agent_locations.shape[0]):
            distance_to_goal = calculate_minimum_distance(
                (current_agent_locations[i, 0], current_agent_locations[i, 1]),
                goal_agent_locations[i],
                map_info
            )
            left_distance = calculate_minimum_distance(
                (current_agent_locations[i, 0]-1, current_agent_locations[i, 1]),
                goal_agent_locations[i],
                map_info
            ) - distance_to_goal
            right_distance = calculate_minimum_distance(
                (current_agent_locations[i, 0]+1, current_agent_locations[i, 1]),
                goal_agent_locations[i],
                map_info
            ) - distance_to_goal    
            down_distance = calculate_minimum_distance(
                (current_agent_locations[i, 0], current_agent_locations[i, 1]-1),
                goal_agent_locations[i],
                map_info
            ) - distance_to_goal
            up_distance = calculate_minimum_distance(
                (current_agent_locations[i, 0], current_agent_locations[i, 1]+1),
                goal_agent_locations[i],
                map_info
            ) - distance_to_goal


            if left_distance > 0 and right_distance > 0:
                dx = 0
            elif left_distance > 0 and right_distance < 0:
                dx = -1
            elif left_distance < 0 and right_distance > 0:
                dx = 1
            else:
                dx = 1
            if down_distance > 0 and up_distance > 0:
                dy = 0
            elif down_distance > 0 and up_distance < 0:
                dy = -1
            elif down_distance < 0 and up_distance > 0:
                dy = 1
            else:
                dy = 1
            feature[4, current_agent_locations[i, 0], current_agent_locations[i, 1]] = dx
            feature[5, current_agent_locations[i, 0], current_agent_locations[i, 1]] = dy
            feature[6, current_agent_locations[i, 0], current_agent_locations[i, 1]] = distance_to_goal

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


    
    
    

        

    
    
