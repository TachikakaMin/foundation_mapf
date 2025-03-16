from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from tools.utils import read_map, read_distance_map, parse_file_name, construct_input_feature
from concurrent.futures import ThreadPoolExecutor

class MAPFDataset(Dataset):

    def __init__(self, input_files, feature_dim, feature_type, first_step=False):
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        self.input_files = input_files
        # file_indices 将直接存储文件名的字节字符串
        self.file_indices = None
        self.step_indices = None
        self.load_file_info(first_step)

    def load_file_info(self, first_step):
        print("Loading file information...")
        # 使用普通列表收集数据
        file_names = []
        step_idxs = []
        if first_step:
            for file_name in self.input_files:
                file_names.extend([file_name.encode('utf-8')])
                step_idxs.extend([0])
        else:
            with ThreadPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(self.load_single_file_info, self.input_files),
                    total=len(self.input_files),
                    desc="Scanning files"
                ))
        
            # 先收集所有数据
            for result in results:
                if result is not None:
                    file_name, steps = result
                    file_names.extend([file_name.encode('utf-8')] * steps)
                    step_idxs.extend(range(steps))
        
        # 最后一次性转换为numpy数组
        self.file_indices = np.array(file_names, dtype=np.bytes_)
        self.step_indices = np.array(step_idxs, dtype=np.int32)
        
        print(f"Found {len(self.input_files)} files with total {len(self.file_indices)} steps")

    def load_single_file_info(self, file_name):
        """Helper function to load a single file's metadata"""
        try:
            with open(file_name, "rb") as f:
                steps = np.int64(np.frombuffer(f.read(2), dtype=np.int16)[0])
                return file_name, steps
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            return None

    def __getitem__(self, idx):
        file_name = self.file_indices[idx].decode('utf-8')
        step_idx = self.step_indices[idx]
        
        with open(file_name, "rb") as f:
            steps = np.int64(np.frombuffer(f.read(2), dtype=np.int16)[0])
            agent_num = np.int64(np.frombuffer(f.read(2), dtype=np.int16)[0])
            step_data_size = agent_num * 3
            
            # Read only the required step's data
            f.seek(4 + step_data_size * step_idx)
            step_data = np.frombuffer(f.read(step_data_size), dtype=np.uint8)
            
            agent_locations = step_data[:2*agent_num].reshape(agent_num, 2)
            actions = step_data[2*agent_num:3*agent_num]
            
            # Read goal locations (last step)
            f.seek(4 + step_data_size * (steps - 1))
            last_step_data = np.frombuffer(f.read(step_data_size), dtype=np.uint8)
            goal_locations = last_step_data[:2*agent_num].reshape(agent_num, 2)

        map_name, _ = parse_file_name(file_name)
        # 按需加载map和distance_map
        map_data = read_map(map_name)
        distance_map = read_distance_map(map_name)
        
        # Convert to tensors
        agent_locations = torch.tensor(agent_locations, dtype=torch.long)
        goal_locations = torch.tensor(goal_locations, dtype=torch.long)
        actions = torch.tensor(actions, dtype=torch.long)

        input_features = construct_input_feature(
            map_data,
            agent_locations,
            goal_locations,
            distance_map,
            self.feature_dim,
            self.feature_type
        )

        output_features = torch.zeros(map_data.shape, dtype=torch.long)
        output_features[agent_locations[:, 0], agent_locations[:, 1]] = actions

        mask = torch.zeros(map_data.shape, dtype=torch.uint8)
        mask[agent_locations[:, 0], agent_locations[:, 1]] = 1
        # map_data = torch.zeros((32,32), dtype=torch.float32)
        # input_features = torch.zeros((self.feature_dim, map_data.shape[0], map_data.shape[1]), dtype=torch.float32)
        # output_features = torch.zeros((map_data.shape[0], map_data.shape[1]), dtype=torch.long)
        # mask = torch.zeros((map_data.shape[0], map_data.shape[1]), dtype=torch.uint8)
        return {
            "feature": input_features,
            "action": output_features,
            "mask": mask,
            "file_name": file_name,
        }

    def __len__(self):
        return len(self.file_indices)