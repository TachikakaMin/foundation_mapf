import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import yaml
from yaml import CLoader, Loader
import orjson
from tqdm import tqdm
import h5py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


class MAPFDataset(Dataset):
    def __init__(self, h5_files, agent_dim):
        self.agent_dim = agent_dim
        self.h5_files = h5_files
        self.train_data_len = []
        self.train_data_map_name = []
        self.train_data_agent_locations = []
        self.all_map_data = {}
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)  # Create cache directory if it doesn't exist
        self.parallel_load_data(self.h5_files)
        
        self.train_data_len = np.array(self.train_data_len, dtype=int)
        self.train_cumsum_len = np.cumsum(self.train_data_len)
        
    def parallel_load_data(self, h5_files):
        """
        Parallelly loads multiple H5 files and processes their training data and action data.

        Args:
        h5_files (list): A list containing all paths to .h5 files
        """
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
                    # 如果文件不合法或有错误，则删除该文件
                    invalid_file = futures[future]
                    print(f"Invalid file detected and removed: {invalid_file}")
                    os.remove(invalid_file)

    def process_h5_file(self, h5_file):
        # Create cache filename based on h5_file path
        cache_file = os.path.join(self.cache_dir, f"{os.path.basename(h5_file)}.npy")
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                cached_data = np.load(cache_file)
                with h5py.File(h5_file, "r") as f:
                    map_name = f['/statistics'].attrs['map']
                    if not (map_name in self.all_map_data.keys()):
                        self.all_map_data[map_name] = torch.FloatTensor(self.read_map(map_name))
                return map_name, cached_data, h5_file
            except Exception as e:
                print(f"Cache load failed for {h5_file}, processing file: {e}")
                
        # If cache doesn't exist or is invalid, process the file
        with h5py.File(h5_file, "r") as f:
            map_name = f['/statistics'].attrs['map']
            if not (map_name in self.all_map_data.keys()):
                self.all_map_data[map_name] = torch.FloatTensor(self.read_map(map_name))
            agent_locations = self.preprocess_h5_data(f)
            
            # Save to cache
            try:
                np.save(cache_file, agent_locations)
            except Exception as e:
                print(f"Failed to save cache for {h5_file}: {e}")
                
        return map_name, agent_locations, h5_file

    
    def read_map(self, map_name):
        map_path = f"map_file/{map_name}"
        map_data = []
        with open(map_path, "r") as f:
            data = f.readlines()
            for l in data[4:]:
                tmp = l.strip()
                tmp = tmp.replace(".", "0").replace("@", "1")  # 1 表示障碍物，0 表示空白
                map_data.append(tmp)
        map_data = np.array([list(line) for line in map_data], dtype=int)
        return map_data
    
    def preprocess_h5_data(self, h5_file):
        agent_names = list(h5_file['/schedule'].keys())
        agent_num = len(agent_names)
        agent_locations = []
        max_length = 0

        for agent_name in agent_names:
            # 读取每个智能体的轨迹数据
            agent_path = h5_file[f'/schedule/{agent_name}/trajectory'][:]
            agent_locs = agent_path.tolist()
            max_length = max(max_length, len(agent_locs))
            agent_locations.append(agent_locs)

        # 对所有智能体的轨迹进行填充，使其长度相同
        for i in range(agent_num):
            sublist = agent_locations[i]
            if len(sublist) < max_length:
                last_x, last_y, last_t = sublist[-1]
                additional_steps = [[last_x, last_y, last_t + j + 1] for j in range(max_length - len(sublist))]
                agent_locations[i] += additional_steps

        agent_locations = np.array(agent_locations, dtype=int)
        return agent_locations
    
    def new_generate_train_data_one(self, agent_locations, map_info, idx):
        m, n = map_info.shape[0], map_info.shape[1]
        # get feature, dim=5, dim 0 is map, dim 1 is current_robot_id, dim 2 is goal_robot_id, dim 3 is distance to goal on x-axis, dim 4 is distance to goal on y-axis
        feature = torch.zeros((3, m, n))
        feature[0] = map_info
        current_agent_locations = agent_locations[:, idx, :2]
        goal_agent_locations = agent_locations[:, -1, :2]

        for i in range(current_agent_locations.shape[0]):
            feature[1, current_agent_locations[i, 0], current_agent_locations[i, 1]] = i+1
            feature[2, goal_agent_locations[i, 0], goal_agent_locations[i, 1]] = i+1
            # feature[3, current_agent_locations[i, 0], current_agent_locations[i, 1]] = goal_agent_locations[i, 0] - current_agent_locations[i, 0]
            # feature[4, current_agent_locations[i, 0], current_agent_locations[i, 1]] = goal_agent_locations[i, 1] - current_agent_locations[i, 1]
        
        # get action info
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
        """_summary_

        Args:
            idx (_type_): time step index

        Returns:
            _type_: _description_
        """
        data_index = np.searchsorted(self.train_cumsum_len, idx, side='right')
        if data_index != 0:
            diff = idx - self.train_cumsum_len[data_index - 1]
        else:
            diff = idx  
        map_name = self.train_data_map_name[data_index]
        map_data = self.all_map_data[map_name]
        agent_locations = self.train_data_agent_locations[data_index]
        try:
            feature, action_info, mask = self.new_generate_train_data_one(agent_locations, map_data, diff)
        except Exception as e:
            print(e, map_name, data_index, diff)
            raise e
        ret_data = {"feature": feature.float(), "action": action_info, "mask": mask}
        return ret_data


    
    
    

        

    
    