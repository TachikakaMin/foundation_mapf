from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from tools.utils import read_map
from tools.cached_distance_reader import read_distance_map_cached
from tools.extensions import construct_input_feature_cpp as cpp_features
import struct
import os
import glob


class MAPFDataset(Dataset):
    """Lazy loading dataset for .mbin files - only loads file metadata, not all steps"""

    def __init__(self, input_files, feature_dim, feature_type, first_step=False):
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        self.input_files = input_files
        self.first_step = first_step
        self._map_name_cache = {}
        self._map_data_cache = {}
        self._active_file_path = None
        self._active_file_handle = None
        self._last_goal_cache_key = None
        self._last_goal_cache_value = None

        # 只存储文件级别的元数据
        self.file_metadata = []  # [(file_path, num_scenarios, [(scenario_idx, steps, offset, data_size, agent_num)])]
        self.cumulative_steps = []  # 累积 step 数量，用于快速定位

        self.load_file_metadata()

    def load_file_metadata(self):
        """只加载文件级别的元数据，不展开所有 step"""
        print(f"Loading metadata from {len(self.input_files)} files...")

        total_steps = 0

        for file_path in tqdm(self.input_files, desc="Scanning files"):
            if self.first_step:
                # first_step 模式：每个文件只取第一个 scenario 的第一个 step，
                # 但仍保留真实的 scenario 元数据，保证能正确读取最终 goal。
                scenarios_info = self._read_file_header(file_path)
                if not scenarios_info:
                    raise RuntimeError(f"空的 .mbin 文件，无法读取 first_step 样本: {file_path}")
                first_scenario = scenarios_info[0]
                self.file_metadata.append((file_path, 1, [first_scenario]))
                self.cumulative_steps.append(total_steps)
                total_steps += 1
            else:
                # 正常模式：读取文件的 scenario 元数据
                scenarios_info = self._read_file_header(file_path)
                num_scenarios = len(scenarios_info)

                # 计算这个文件的总 step 数
                file_steps = sum(steps for _, steps, _, _, _ in scenarios_info)

                self.file_metadata.append((file_path, num_scenarios, scenarios_info))
                self.cumulative_steps.append(total_steps)
                total_steps += file_steps

        self.total_steps = total_steps
        print(f"Found {len(self.input_files)} files with total {self.total_steps} steps")

    def _read_file_header(self, file_path):
        """读取单个 .mbin 文件的头部信息（不读取实际数据）"""
        scenarios_info = []

        with open(file_path, "rb") as f:
            # 读取文件头部
            header_data = f.read(16)  # MergedFileHeader size
            num_scenarios = struct.unpack('I', header_data[:4])[0]

            # 读取索引表
            for scenario_idx in range(num_scenarios):
                index_data = f.read(272)  # ScenarioIndex size: 8+4+2+2+256
                offset = struct.unpack('Q', index_data[:8])[0]
                data_size = struct.unpack('I', index_data[8:12])[0]
                steps = struct.unpack('H', index_data[12:14])[0]
                agent_num = struct.unpack('H', index_data[14:16])[0]

                scenarios_info.append((scenario_idx, steps, offset, data_size, agent_num))

        return scenarios_info

    def _locate_step(self, global_idx):
        """根据全局 step 索引，定位到具体的文件、scenario 和 step"""
        # 二分查找定位文件
        left, right = 0, len(self.cumulative_steps) - 1
        file_idx = 0

        while left <= right:
            mid = (left + right) // 2
            if self.cumulative_steps[mid] <= global_idx:
                file_idx = mid
                left = mid + 1
            else:
                right = mid - 1

        # 计算文件内的相对索引
        file_start = self.cumulative_steps[file_idx]
        relative_idx = global_idx - file_start

        # 在文件内定位 scenario 和 step
        file_path, num_scenarios, scenarios_info = self.file_metadata[file_idx]

        current_offset = 0
        for scenario_idx, steps, offset, data_size, agent_num in scenarios_info:
            if relative_idx < current_offset + steps:
                step_idx = relative_idx - current_offset
                return file_path, scenario_idx, step_idx, steps, offset, data_size, agent_num
            current_offset += steps

        raise IndexError(f"Invalid index: {global_idx}")

    def parse_map_name_from_mbin(self, file_name):
        """从.mbin文件名解析地图名称"""
        cached = self._map_name_cache.get(file_name)
        if cached is not None:
            return cached

        # 文件名格式：maze-32-32-30-4-85-0-128.mbin
        basename = os.path.basename(file_name).replace('.mbin', '')
        parts = basename.split('-')

        if len(parts) >= 8:
            # 地图模式：maze-32-32-30-4-85 (前6个部分)
            # 地图ID：0 (第7个部分)
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
            # 回退方案：从目录结构推断
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

        self._map_name_cache[file_name] = map_name
        return map_name

    def _get_file_handle(self, file_path):
        if self._active_file_path != file_path or self._active_file_handle is None:
            if self._active_file_handle is not None:
                self._active_file_handle.close()
            self._active_file_handle = open(file_path, "rb")
            self._active_file_path = file_path
        return self._active_file_handle

    def _get_goal_locations(self, file_handle, file_path, scenario_idx, offset, step_data_size, steps, agent_num):
        cache_key = (file_path, scenario_idx)
        if self._last_goal_cache_key == cache_key and self._last_goal_cache_value is not None:
            return self._last_goal_cache_value

        last_step_offset = offset + 4 + step_data_size * (steps - 1)
        file_handle.seek(last_step_offset)
        last_step_data = file_handle.read(step_data_size)
        last_step_array = np.frombuffer(last_step_data, dtype=np.uint8)
        goal_locations = last_step_array[:2 * agent_num].reshape(agent_num, 2).copy()
        self._last_goal_cache_key = cache_key
        self._last_goal_cache_value = goal_locations
        return goal_locations

    def __getitem__(self, idx):
        # 定位到具体的文件、scenario 和 step
        file_path, scenario_idx, step_idx, steps, offset, data_size, agent_num = self._locate_step(idx)

        # 读取指定的 step 数据
        file_handle = self._get_file_handle(file_path)
        step_data_size = agent_num * 3
        step_offset = offset + 4 + step_data_size * step_idx
        file_handle.seek(step_offset)
        step_data = file_handle.read(step_data_size)

        step_array = np.frombuffer(step_data, dtype=np.uint8)
        agent_locations = step_array[:2*agent_num].reshape(agent_num, 2)
        actions = step_array[2*agent_num:3*agent_num]
        goal_locations = self._get_goal_locations(
            file_handle,
            file_path,
            scenario_idx,
            offset,
            step_data_size,
            steps,
            agent_num,
        )

        # 解析地图信息
        map_name = self.parse_map_name_from_mbin(file_path)
        map_data = self._map_data_cache.get(map_name)
        if map_data is None:
            map_data = read_map(map_name).astype(np.float32, copy=False)
            self._map_data_cache[map_name] = map_data
        distance_map = read_distance_map_cached(map_name)

        # 转换为tensors
        agent_locations = torch.tensor(agent_locations, dtype=torch.long)
        goal_locations = torch.tensor(goal_locations, dtype=torch.long)
        actions = torch.tensor(actions, dtype=torch.long)

        # 转换为numpy数组以便C++处理
        map_data_np = map_data
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
            "file_name": file_path,
        }

    def __len__(self):
        return self.total_steps

    def __del__(self):
        if getattr(self, "_active_file_handle", None) is not None:
            self._active_file_handle.close()
