from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from tools.utils import read_map
from tools.cached_distance_reader import read_distance_map_cached
from tools.extensions import construct_input_feature_cpp as cpp_features
from concurrent.futures import ThreadPoolExecutor
import struct
import os
import glob

class MAPFDataset(Dataset):
    """专门处理.mbin合并文件的数据集"""

    def __init__(self, input_files, feature_dim, feature_type, first_step=False):
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        self.input_files = input_files
        self.file_indices = None
        self.step_indices = None
        self.scenario_indices = None
        self.load_file_info(first_step)

    def load_file_info(self, first_step):
        print("Loading merged file information...")
        
        file_names = []
        step_idxs = []
        scenario_idxs = []
        
        if first_step:
            for file_name in self.input_files:
                file_names.append(file_name.encode('utf-8'))
                step_idxs.append(0)
                scenario_idxs.append(0)
        else:
            with ThreadPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(self.load_single_merged_file_info, self.input_files),
                    total=len(self.input_files),
                    desc="Scanning merged files"
                ))
            
            # 收集所有数据
            for result in results:
                if result is not None:
                    file_name, scenarios_info = result
                    for scenario_idx, steps in scenarios_info:
                        for step in range(steps):
                            file_names.append(file_name.encode('utf-8'))
                            step_idxs.append(step)
                            scenario_idxs.append(scenario_idx)
        
        # 转换为numpy数组
        self.file_indices = np.array(file_names, dtype=np.bytes_)
        self.step_indices = np.array(step_idxs, dtype=np.int32)
        self.scenario_indices = np.array(scenario_idxs, dtype=np.int32)
        
        print(f"Found {len(self.input_files)} merged files with total {len(self.file_indices)} steps")

    def load_single_merged_file_info(self, file_name):
        """加载单个合并文件的元数据"""
        with open(file_name, "rb") as f:
            # 读取文件头部
            header_data = f.read(16)  # MergedFileHeader size
            num_scenarios = struct.unpack('I', header_data[:4])[0]
            
            scenarios_info = []
            
            # 读取索引表
            for scenario_idx in range(num_scenarios):
                index_data = f.read(272)  # ScenarioIndex size: 8+4+2+2+256
                steps = struct.unpack('H', index_data[12:14])[0]
                scenarios_info.append((scenario_idx, steps))
            
            return file_name, scenarios_info

    def parse_map_name_from_mbin(self, file_name):
        """从.mbin文件名解析地图名称"""
        # 文件名格式: maze-32-32-30-4-85-0-128.mbin
        basename = os.path.basename(file_name).replace('.mbin', '')
        parts = basename.split('-')
        
        if len(parts) >= 8:
            # 地图模式: maze-32-32-30-4-85 (前6个部分)
            # 地图ID: 0 (第7个部分)
            map_pattern = '-'.join(parts[:6])  # maze-32-32-30-4-85
            map_id = parts[6]  # 0
            map_name = f"data/map_files/{map_pattern}/{map_pattern}-{map_id}.map"
            
            # 验证地图文件是否存在
            if not os.path.exists(map_name):
                # 如果直接路径不存在, 尝试查找该目录下的任意地图文件
                map_files = glob.glob(f"data/map_files/{map_pattern}/*.map")
                if map_files:
                    map_name = map_files[0]  # 使用第一个可用的地图文件
                else:
                    raise FileNotFoundError(f"地图文件不存在: {map_name}")
        else:
            # 回退方案: 从目录结构推断
            dir_parts = file_name.split('/')
            for part in dir_parts:
                if part.startswith('maze-') and part.count('-') >= 4:
                    # 使用第一个可用的地图文件
                    map_files = glob.glob(f"data/map_files/{part}/*.map")
                    if map_files:
                        map_name = map_files[0]
                    else:
                        raise FileNotFoundError(f"无法找到地图文件: {part}")
                    break
            else:
                raise FileNotFoundError(f"无法解析地图名称: {file_name}")
        
        return map_name

    def __getitem__(self, idx):
        file_name = self.file_indices[idx].decode('utf-8')
        step_idx = self.step_indices[idx]
        scenario_idx = self.scenario_indices[idx]
        
        with open(file_name, "rb") as f:
            # 读取文件头部
            header_data = f.read(16)
            num_scenarios = struct.unpack('I', header_data[:4])[0]
            
            # 读取索引表找到目标场景
            index_table_start = 16
            scenario_index_size = 272  # 8+4+2+2+256 = 272字节
            
            f.seek(index_table_start + scenario_idx * scenario_index_size)
            index_data = f.read(scenario_index_size)
            
            offset = struct.unpack('Q', index_data[:8])[0]
            data_size = struct.unpack('I', index_data[8:12])[0]
            steps = struct.unpack('H', index_data[12:14])[0]
            agent_num = struct.unpack('H', index_data[14:16])[0]
            
            # 计算步骤数据大小
            step_data_size = agent_num * 3
            
            # 读取指定步骤的数据
            step_offset = offset + 4 + step_data_size * step_idx
            f.seek(step_offset)
            step_data = f.read(step_data_size)
            
            step_array = np.frombuffer(step_data, dtype=np.uint8)
            agent_locations = step_array[:2*agent_num].reshape(agent_num, 2)
            actions = step_array[2*agent_num:3*agent_num]
            
            # 读取目标位置（最后一步）
            last_step_offset = offset + 4 + step_data_size * (steps - 1)
            f.seek(last_step_offset)
            last_step_data = f.read(step_data_size)
            last_step_array = np.frombuffer(last_step_data, dtype=np.uint8)
            goal_locations = last_step_array[:2*agent_num].reshape(agent_num, 2)

        # 解析地图信息
        map_name = self.parse_map_name_from_mbin(file_name)
        map_data = read_map(map_name)
        distance_map = read_distance_map_cached(map_name)
        
        # 转换为tensors
        agent_locations = torch.tensor(agent_locations, dtype=torch.long)
        goal_locations = torch.tensor(goal_locations, dtype=torch.long)
        actions = torch.tensor(actions, dtype=torch.long)

        
        # 转换为numpy数组以便C++处理
        map_data_np = map_data.astype(np.float32) if isinstance(map_data, np.ndarray) else np.array(map_data, dtype=np.float32)
        agent_locations_np = agent_locations.cpu().numpy().astype(np.int64)
        goal_locations_np = goal_locations.cpu().numpy().astype(np.int64)
        
        # 调用C++函数
        input_features_np = cpp_features(
            map_data_np,
            agent_locations_np, 
            goal_locations_np,
            distance_map,
            self.feature_dim,
            self.feature_type
        )
        
        # 转换回torch tensor
        input_features = torch.from_numpy(input_features_np).to(agent_locations.device)


        output_features = torch.zeros(map_data.shape, dtype=torch.long)
        output_features[agent_locations[:, 0], agent_locations[:, 1]] = actions

        mask = torch.zeros(map_data.shape, dtype=torch.uint8)
        mask[agent_locations[:, 0], agent_locations[:, 1]] = 1
        
        return {
            "feature": input_features,
            "action": output_features,
            "mask": mask,
            "file_name": file_name,
        }

    def __len__(self):
        return len(self.file_indices) 
    