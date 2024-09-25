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

dx = [0, 0, 1, -1, 0]
dy = [1, -1, 0, 0, 0]

class MAPFDataset(Dataset):
    def __init__(self, data_path, agent_dim):
        self.agent_dim = agent_dim
        self.data_path = data_path
        if os.path.isdir(data_path):
            yaml_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".yaml")]
        else:
            yaml_files = [self.data_path]
        
        self.train_data, self.action_data = [], []
        for yaml_file in tqdm(yaml_files):
            with open(yaml_file, "rb") as f:
                raw_data = yaml.load(f, Loader=CLoader)
            map_name = raw_data['statistics']['map']
            map_data = self.read_map(map_name)
            agent_num, agent_locations = self.preprocess_data(raw_data)
            
            # (n,n, 1), (n, n, agent_dim), (t, n, n, agent_dim)
            train_data, action_data = self.generate_train_data(agent_num, agent_locations, map_data)
            self.train_data.append(train_data)
            self.action_data.append(action_data)
        self.train_data = torch.cat(self.train_data, dim=0)
        self.action_data = torch.cat(self.action_data, dim=0)
        
        
    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        train_data = self.train_data[idx].permute((2, 0, 1))
        action_info = self.action_data[idx]
        mask = action_info.any(-1)
        action_info = action_info.argmax(-1).long()
        ret_data = {"feature": train_data.float(), "action": action_info, "mask": mask}
        return ret_data

    def get_action_info(self, frame, next_frame, agent_num):
        shift_left = torch.zeros_like(frame)
        shift_right = torch.zeros_like(frame)
        shift_up = torch.zeros_like(frame)
        shift_down = torch.zeros_like(frame)
        
        # Left shift
        shift_left[:, :-1, :] = frame[:, 1:, :]
        # Right shift
        shift_right[:, 1:, :] = frame[:, :-1, :]
        # Up shift
        shift_up[:-1, :, :] = frame[1:, :, :]
        # Down shift
        shift_down[1:, :, :] = frame[:-1, :, :]
        
        not_shift = frame
        action_info = []
        check_list = [shift_left, shift_right, shift_up, shift_down, not_shift]
        mask_next_frame = torch.any(next_frame != 0, dim=-1)
        for mx in check_list:
            mask_mx = torch.any(mx != 0, dim=-1)
            comparison = torch.all(mx == next_frame, axis=-1) & mask_mx & mask_next_frame
            action_info.append(comparison)

        action_info = torch.stack(action_info, dim=-1).float()
        assert(action_info.sum() == agent_num)
        return action_info
        
        
        

    def generate_train_data(self, agent_num, agent_locations, map_data):
        size_n, size_m = map_data.shape
        max_time = agent_locations.shape[1]
        
        map_info = np.expand_dims(map_data, axis=-1) # (n,m,1)
        map_info = torch.FloatTensor(map_info)
        train_data = []
        
        t = max_time-1
        goal_loc_info = np.zeros((size_n, size_m, self.agent_dim), dtype=int)
        indices = np.arange(agent_num)
        binary_strings = np.array([list(format(i+1, f'0{self.agent_dim}b')) for i in indices], dtype=int)
        agent_data = agent_locations[:,t,:][:, :2]
        goal_loc_info[agent_data[:, 0], agent_data[:, 1]] = binary_strings
        goal_loc_info = torch.FloatTensor(goal_loc_info)
        
        for t in range(max_time):
            current_loc_info = np.zeros((size_n, size_m, self.agent_dim), dtype=int)
            indices = np.arange(agent_num)
            binary_strings = np.array([list(format(i+1, f'0{self.agent_dim}b')) for i in indices], dtype=int)
            agent_data = agent_locations[:,t,:][:, :2]
            current_loc_info[agent_data[:, 0], agent_data[:, 1]] = binary_strings
            current_loc_info = torch.FloatTensor(goal_loc_info)
            feature = [map_info, goal_loc_info, current_loc_info] 
            feature = torch.cat(feature, dim=-1)
            train_data.append(feature)
        train_data = torch.stack(train_data)
        action_data = []
        for t in range(max_time-1):
            train_data_x = train_data[t][:,:,self.agent_dim+1:]
            train_data_x_next = train_data[t+1][:,:,self.agent_dim+1:]
            action_info = self.get_action_info(train_data_x, train_data_x_next, agent_num)
            action_data.append(action_info)
            
        action_data = torch.stack(action_data)
        return train_data[:-1], action_data

    def read_map(self, map_name):
        map_path = f"map_file/{map_name}"
        ret_data = []
        with open(map_path, "r") as f:
            data = f.readlines()
            for l in data[4:]:
                tmp = l[:-1]
                tmp = tmp.replace(".", "1").replace("@", "0")
                ret_data.append(tmp)
        ret_data = np.array([list(line) for line in ret_data], dtype=int)
        return ret_data

    def preprocess_data(self, raw_data):
        agent_num = len(raw_data["schedule"])
        agent_locations = []
        max_length = 0
        for i, agent_name in enumerate(raw_data["schedule"]):
            agent_locs = []
            agent_path = raw_data["schedule"][agent_name]
            for loc in agent_path:
                x, y, t = loc['x'], loc['y'], loc['t']
                agent_locs.append([x, y, t])
            max_length = max(max_length, len(agent_locs))
            agent_locations.append(agent_locs)
        for i in range(agent_num):
            sublist = agent_locations[i]
            if len(sublist) < max_length:
                # 用最后一个元素填充
                agent_locations[i] += [sublist[-1]] * (max_length - len(sublist))
        agent_locations = np.array(agent_locations, dtype=int)
        return agent_num, agent_locations