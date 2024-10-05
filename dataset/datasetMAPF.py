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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# five actions the agent can take:[left, right, up, down, stay]->[[0,-1],[0,1],[1,0],[-1,0],[0,0]]
dx = [0, 0, 1, -1, 0]
dy = [1, -1, 0, 0, 0]



class MAPFDataset(Dataset):
    def __init__(self, data_path, agent_dim):
        self.agent_dim = agent_dim
        self.data_path = data_path
        
        # .npz are preprocessed data. 
        # just load and convert them into pytorch tensors.
        if ".npz" in self.data_path:
            self.load(self.data_path)
            return
        
        if os.path.isdir(data_path):
            # A list containing the paths of all .yaml files.
            yaml_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".yaml")]
        else:
            yaml_files = [self.data_path]
        
        self.parallel_load_data(yaml_files)
        self.train_data = torch.cat(self.train_data, dim=0)  # 沿第0维进行拼接  #(t, n, m, feature_dim)
        self.action_data = torch.cat(self.action_data, dim=0)
        self.save("data")
    
    def load(self, load_path):
        data = np.load(load_path)
        self.train_data = torch.from_numpy(data['train_data'])
        self.train_data = torch.FloatTensor(self.train_data)
        self.action_data = torch.from_numpy(data['action_data'])
        self.action_data = torch.FloatTensor(self.action_data)
        
        
    def parallel_load_data(self, yaml_files):
        """
        Parallelly loads multiple YAML files and processes their training data and action data.

        Args:
        yaml_files (list): A list containing all paths to .yaml files
        """
        self.train_data, self.action_data = [], []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit tasks to the executor for each yaml file
            futures = {executor.submit(self.process_yaml_file, yaml_file): yaml_file for yaml_file in yaml_files}
            
            for future in tqdm(as_completed(futures), total=len(yaml_files)):
                # 这里使用 tqdm 库显示进度条，跟踪任务的完成情况
                train_data, action_data = future.result()
                self.train_data.append(train_data)  # (number of samples, max_time, n, m, feature_dim)
                self.action_data.append(action_data) # (number of samples, max_time-1, n, m, 5)
                
    # Function to process each YAML file
    def process_yaml_file(self, yaml_file):
        with open(yaml_file, "rb") as f:
            raw_data = yaml.load(f, Loader=CLoader)
        map_name = raw_data['statistics']['map']
        map_data = self.read_map(map_name)
        agent_num, agent_locations = self.preprocess_data(raw_data)
        
        # (max_time, n, m, feature_dim) (max_time-1, n, m, 5)
        train_data, action_data = self.generate_train_data(agent_num, agent_locations, map_data)
        
        return train_data, action_data
    
    def read_map(self, map_name):
        map_path = f"map_file/{map_name}"
        map_data = []                  
        with open(map_path, "r") as f:
            data = f.readlines()
            for l in data[4:]:
                tmp = l[:-1]
                tmp = tmp.replace(".", "0").replace("@", "1") # 1 for obstacle, 0 for empty space
                map_data.append(tmp)
        map_data = np.array([list(line) for line in map_data], dtype=int)
        return map_data
    
    def preprocess_data(self, raw_data):
        agent_num = len(raw_data["schedule"])
        agent_locations = []
        # 记录所有agent路径的最大长度
        max_length = 0
        for i, agent_name in enumerate(raw_data["schedule"]):
            agent_locs = []
            agent_path = raw_data["schedule"][agent_name]
            for loc in agent_path:
                x, y, t = loc['x'], loc['y'], loc['t']
                agent_locs.append([x, y, t])
            max_length = max(max_length, len(agent_locs))
            agent_locations.append(agent_locs) # 包含每个智能体路径
        for i in range(agent_num):
            sublist = agent_locations[i] # 获取第 i 个智能体的路径
            if len(sublist) < max_length:
                # 用最后一个元素填充
                # agent_locations[i] += [sublist[-1]] * (max_length - len(sublist))
                # 获取最后一个位置的 x 和 y 值
                last_x, last_y, last_t = sublist[-1]
                # 生成新的填充项，x 和 y 不变，t 从 last_t + 1 开始递增
                additional_steps = [[last_x, last_y, last_t + j + 1] for j in range(max_length - len(sublist))]
                # 将这些填充项添加到原来的路径中
                agent_locations[i] += additional_steps
        agent_locations = np.array(agent_locations, dtype=int)
        return agent_num, agent_locations
    
    def generate_train_data(self, agent_num, agent_locations, map_data):
        size_n, size_m = map_data.shape
        max_time = agent_locations.shape[1]  # shape of agent_locations: (agent_num, time_step, 3)
        
        map_info = np.expand_dims(map_data, axis=-1) # (n,m,1) # 将 map_data 在最后一个维度上进行扩展，形状从 (n, m) 变为 (n, m, 1)
        map_info = torch.FloatTensor(map_info)
        
        goal_loc_info = np.zeros((size_n, size_m, self.agent_dim), dtype=int)
        t = max_time-1
        # 提取了智能体在最后一个时间步的 (x, y) 位置作为目标位置
        agent_data = agent_locations[:,t,:][:, :2]
        indices = np.arange(agent_num)
        # format() 函数，用于将数字 i+1 转换为一个固定长度的二进制字符串，其中 self.agent_dim 指定二进制字符串的长度。
        binary_strings = np.array([list(format(i+1, f'0{self.agent_dim}b')) for i in indices], dtype=int) # shape: (agent_num, agent_dim)
        # 将每个智能体的目标位置填充气自身二进制编码
        goal_loc_info[agent_data[:, 0], agent_data[:, 1]] = binary_strings
        goal_loc_info = torch.FloatTensor(goal_loc_info)
        
        train_data = []
        for t in range(max_time):
            current_loc_info = np.zeros((size_n, size_m, self.agent_dim), dtype=int)
            agent_data = agent_locations[:,t,:][:, :2]
            indices = np.arange(agent_num)
            binary_strings = np.array([list(format(i+1, f'0{self.agent_dim}b')) for i in indices], dtype=int)
            current_loc_info[agent_data[:, 0], agent_data[:, 1]] = binary_strings
            current_loc_info = torch.FloatTensor(current_loc_info)
            
            feature = [map_info, goal_loc_info, current_loc_info]
            # 将地图信息（map_info）、目标位置信息（goal_loc_info）和当前智能体的位置信息（current_loc_info）连接起来，构成当前时间步的特征向量。 
            feature = torch.cat(feature, dim=-1)
            
            train_data.append(feature)
        # 列表转换为 PyTorch 张量，形状为 (max_time, n, m, feature_dim)
        train_data = torch.stack(train_data)
        
        action_data = []
        for t in range(max_time-1):
            train_data_x = train_data[t][:,:,self.agent_dim+1:] # 当前时间步的每个格子的机器人编号
            train_data_x_next = train_data[t+1][:,:,self.agent_dim+1:]
            action_info = self.get_action_info(train_data_x, train_data_x_next, agent_num)
            action_data.append(action_info)
        
        # shape: (max_time-1, n, m, 5)
        action_data = torch.stack(action_data)
        
        return train_data[:-1], action_data
    
    def get_action_info(self, frame, next_frame, agent_num):
        """
        通过比较当前帧（frame）和下一帧（next_frame）中的智能体位置，计算每个智能体在某一时间步采取的动作信息
        返回的 action_info 包含智能体在每个时间步所采取的动作信息
        """
        shift_left = torch.zeros_like(frame)
        shift_right = torch.zeros_like(frame)
        shift_up = torch.zeros_like(frame)
        shift_down = torch.zeros_like(frame)
        
        # Left shift
        shift_left[:, :-1, :] = frame[:, 1:, :] # 将 frame 的所有元素向左移动一列（原来的第一列被丢弃，最后一列填充 0）
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
        # 检测下一个时间点的map中的每个位置是否有智能体（智能体位置不为 0）。
        # 如果某个位置有智能体，则掩码为 True，否则为 False。
        mask_next_frame = torch.any(next_frame != 0, dim=-1)
        for i, mx in enumerate(check_list):
            # 之前假设了所有的格子上的智能体都往同一个方向移动一步，变成了map1。
            # 现在检查这个假设是否成立（即真实的下一个时间点的每个格子的智能体编号是不是和map1一样）
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
        
    def save(self, save_path):
        file_name = "dataset.npz"
        file_path = os.path.join(save_path, file_name)
        np.savez(file_path, train_data=self.train_data.cpu().numpy(), action_data=self.action_data.cpu().numpy())
        
    
        
    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): time step index

        Returns:
            _type_: _description_
        """
        # 重新排列维度，将 (高度, 宽度, 通道数) 变成 (通道数, 高度, 宽度)
        train_data = self.train_data[idx].permute((2, 0, 1))
        # 获取当前帧的 action 信息，形状为 (n, m, 动作维度)
        action_info = self.action_data[idx]
        # 如果某个位置上存在任何动作（即该位置有智能体），则返回 True，否则返回 False。
        mask = action_info.any(-1) # (n, m)
        
        # 创建一个全零张量，用于存储每个位置的动作索引
        action_info_result = torch.zeros_like(mask, dtype=torch.long)
        
        # 在 mask 为 True 的位置执行 argmax(-1)，表示计算动作的最大值索引
        # 然后将结果加 1 # 左：1，右：2，上：3，下：4，停留：5;没有agent:0
        action_info_result[mask] = action_info[mask].argmax(-1) + 1  # 只对 mask 为 True 的位置执行
        
        # # 那么 argmax(-1) 会返回每个位置的动作索引，表示智能体在该位置上选择了哪个动作。表示每个位置的动作索引。
        # action_info = action_info.argmax(-1).long()
        # # mask 感觉不对吧，是需要输入时间步的位置，而不是输出时间步的
        # # action感觉也是，可以看get_action_info
        ret_data = {"feature": train_data.float(), "action": action_info_result, "mask": mask}
        return ret_data


    
    
    

        

    
    