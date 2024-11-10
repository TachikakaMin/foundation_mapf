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
        with h5py.File(h5_file, "r") as f:
            # 读取地图名称
            map_name = f['/statistics'].attrs['map']
            if not (map_name in self.all_map_data.keys()):
                self.all_map_data[map_name] = torch.FloatTensor(self.read_map(map_name))
            agent_locations = self.preprocess_h5_data(f)
        f.close()
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
        map_data = np.expand_dims(map_data, axis=-1)
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
    
    
    def get_feature(self, size_n, size_m, agent_num, agent_locations, idx, map_info, goal_loc_info):
        current_loc_info = torch.zeros((size_n, size_m, self.agent_dim), dtype=torch.int)
        cur_agent_data = agent_locations[:, idx, :][:, :2]
        indices = torch.arange(agent_num)

        # 创建二进制编码的特征
        binary_strings = torch.tensor(
            [[int(bit) for bit in format(i + 1, f'0{self.agent_dim}b')] for i in indices],
            dtype=torch.int
        )
        
        current_loc_info[cur_agent_data[:, 0], cur_agent_data[:, 1]] = binary_strings
        current_loc_info = current_loc_info.float()
        
        
        
        goal_vector_info = np.zeros((size_n, size_m, 2), dtype=int)
        goal_agent_data = agent_locations[:,-1,:][:, :2]
        
        # 计算方向向量并按照每个智能体的到目标位置的距离进行归一化
        vec = goal_agent_data - cur_agent_data  # 当前到目标的方向向量
        distances = np.linalg.norm(vec, axis=1, keepdims=True)  # 欧氏距离
        
        # 为了避免除零，将距离为零的情况设置为 1
        distances[distances == 0] = 1
        unit_vec = vec / distances  # 归一化每个向量以生成单位向量
        
        # 将单位向量填充到 goal_vector_info 矩阵中对应的智能体当前位置
        goal_vector_info[cur_agent_data[:, 0], cur_agent_data[:, 1]] = unit_vec
        goal_vector_info = torch.FloatTensor(goal_vector_info)
        
        
        final_feature = torch.empty((size_n, size_m, map_info.size(-1)+ goal_vector_info.size(-1) + goal_loc_info.size(-1) + self.agent_dim), dtype=torch.float)
        
        final_feature[..., :map_info.size(-1)] = map_info # 1
        final_feature[..., map_info.size(-1):map_info.size(-1)+ goal_vector_info.size(-1)] = goal_vector_info # 1+2
        final_feature[..., map_info.size(-1)+ goal_vector_info.size(-1):\
            map_info.size(-1)+ goal_vector_info.size(-1) + goal_loc_info.size(-1)] = goal_loc_info  # 1+2+agent_dim
        final_feature[..., -self.agent_dim:] = current_loc_info # 1+2+agent_dim+agent_dim
        # 将地图信息(map_info)、目标位置信息(goal_loc_info)和当前智能体的位置信息(current_loc_info)连接起来，构成当前时间步的特征向量。 
        return final_feature
    
    def generate_train_data_one(self, agent_locations, map_info, idx):
        size_n, size_m, _ = map_info.shape
        agent_num = agent_locations.shape[0]
        
        goal_loc_info = np.zeros((size_n, size_m, self.agent_dim), dtype=int)
        goal_agent_data = agent_locations[:,-1,:][:, :2] # 提取了智能体在最后一个时间步的 (x, y) 位置作为目标位置
        indices = np.arange(agent_num) 
        # format() 函数，用于将数字 i+1 转换为一个固定长度的二进制字符串，其中 self.agent_dim 指定二进制字符串的长度。
        binary_strings = np.array([list(format(i+1, f'0{self.agent_dim}b')) for i in indices], dtype=int) # shape: (agent_num, agent_dim)
        # 将每个智能体的目标位置填充气自身二进制编码
        goal_loc_info[goal_agent_data[:, 0], goal_agent_data[:, 1]] = binary_strings
        goal_loc_info = torch.FloatTensor(goal_loc_info)
        
        feature = self.get_feature(size_n, size_m, agent_num, 
                                       agent_locations, idx, map_info, goal_loc_info)
        
        feature2 = self.get_feature(size_n, size_m, agent_num, 
                                       agent_locations, idx+1, map_info, goal_loc_info)
        
        train_data_x = feature[:,:,-self.agent_dim:] # 当前时间步的每个格子的机器人编号
        train_data_x_next = feature2[:,:,-self.agent_dim:]
        action_info = self.get_action_info(train_data_x, train_data_x_next, agent_num)
        
        return feature, action_info
        
    
    def get_action_info(self, frame, next_frame, agent_num):
        """
        通过比较当前帧(frame)和下一帧(next_frame)中的智能体位置，计算每个智能体在某一时间步采取的动作信息
        返回的 action_info 包含智能体在每个时间步所采取的动作信息
        """
        shift_left = torch.zeros_like(frame)
        shift_right = torch.zeros_like(frame)
        shift_up = torch.zeros_like(frame)
        shift_down = torch.zeros_like(frame)
        
        # Left shift
        shift_left[:, :-1, :] = frame[:, 1:, :] # 将 frame 的所有元素向左移动一列(原来的第一列被丢弃，最后一列填充 0)
        # Right shift
        shift_right[:, 1:, :] = frame[:, :-1, :]
        # Up shift
        shift_up[:-1, :, :] = frame[1:, :, :]
        # Down shift
        shift_down[1:, :, :] = frame[:-1, :, :]
        # stay
        not_shift = frame
        
        action_info = []
        check_list = [shift_left, shift_right, shift_up, shift_down, not_shift]
        # 检测下一个时间点的map中的每个位置是否有智能体(智能体位置不为 0)。
        # 如果某个位置有智能体，则掩码为 True，否则为 False。
        mask_next_frame = torch.any(next_frame != 0, dim=-1)
        for i, mx in enumerate(check_list):
            # 之前假设了所有的格子上的智能体都往同一个方向移动一步，变成了map1。
            # 现在检查这个假设是否成立(即真实的下一个时间点的每个格子的智能体编号是不是和map1一样)
            mask_mx = torch.any(mx != 0, dim=-1)
            # 比较当前移动方向 mx 和 next_frame，检查是否有智能体移动到了 next_frame 中的某个位置；并且确保这些位置确实有智能体并且是有效的动作            
            comparison = torch.all(mx == next_frame, axis=-1) & mask_mx & mask_next_frame
            
            # 将方向结果移回到原来的位置，使得动作信息反映的是当前时间点采取的动作
            if i == 0:  # origin left shift, now shift right
                comparison = torch.cat([torch.zeros_like(comparison[:, :1]), comparison[:, :-1]], dim=1)
            elif i == 1:  # origin right shift
                comparison = torch.cat([comparison[:, 1:], torch.zeros_like(comparison[:, :1])], dim=1)
            elif i == 2:  # origin up shift
                comparison =  torch.cat([torch.zeros_like(comparison[:1, :]), comparison[:-1, :]], dim=0)
            elif i == 3:  # origin down shift
                comparison = torch.cat([comparison[1:, :], torch.zeros_like(comparison[:1, :])], dim=0)
            
            action_info.append(comparison)

        # 生成一个包含所有动作信息的张量。张量的形状为 (n, m, 5)
        action_info = torch.stack(action_info, dim=-1).float()
        # 检查生成的 action_info 中的所有动作信息的总和是否等于智能体数量
        assert(action_info.sum() == agent_num)
        return action_info
    
        
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
        diff = idx
        if data_index != 0:
            diff = idx - self.train_cumsum_len[data_index - 1]
        
        map_name = self.train_data_map_name[data_index]
        map_data = self.all_map_data[map_name]
        agent_locations = self.train_data_agent_locations[data_index]
        
        try:
            train_data, action_info = self.generate_train_data_one(agent_locations, map_data, diff)
        except Exception as e:
            print(e, map_name, data_index, diff)
        # 重新排列维度，将 (高度, 宽度, 通道数) 变成 (通道数, 高度, 宽度)
        train_data = train_data.permute((2, 0, 1))
        # 获取当前帧的 action 信息，形状为 (n, m, 动作维度)
        action_info = action_info
        # 如果某个位置上存在任何动作(即该位置有智能体)，则返回 True，否则返回 False。
        mask = action_info.any(-1) # (n, m)
        
        # 创建一个全零张量，用于存储每个位置的动作索引
        action_info_result = torch.zeros_like(mask, dtype=torch.long)
        
        # 在 mask 为 True 的位置执行 argmax(-1)，表示计算动作的最大值索引
        # 然后将结果加 1 # 左：0，右：1，上：2，下：3，停留：4;没有agent:0  # 后面会mask掉,所以两个都可以是0
        action_info_result[mask] = action_info[mask].argmax(-1)   # 只对 mask 为 True 的位置执行
        
        # # 那么 argmax(-1) 会返回每个位置的动作索引，表示智能体在该位置上选择了哪个动作。表示每个位置的动作索引。
        # action_info = action_info.argmax(-1).long()
        # # mask 感觉不对吧，是需要输入时间步的位置，而不是输出时间步的
        # # action感觉也是，可以看get_action_info
        ret_data = {"feature": train_data.float(), "action": action_info_result, "mask": mask}
        return ret_data


    
    
    

        

    
    