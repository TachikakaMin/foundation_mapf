import torch
import numpy as np
from torch.utils.data import Dataset
from tools.utils import read_map, read_distance_map, parse_file_name, construct_input_feature

class MAPFDataset(Dataset):

    def __init__(self, input_files, feature_dim, feature_type):
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        self.input_files = input_files
        self.input_data = []
        self.maps = {}
        self.distance_maps = {}

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        file_name = self.input_files[idx]
        data = self.process_input_file(file_name)
        map_name, _, _ = parse_file_name(file_name)
        map_data = self.get_map(map_name)
        distance_map = self.get_distance_map(map_name)
        agent_locations = torch.tensor(data["agent_locations"], dtype=torch.long)
        goal_locations = torch.tensor(data["goal_locations"], dtype=torch.long)
        actions = torch.tensor(data["actions"], dtype=torch.long)

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

        return {
            "feature": input_features,
            "action": output_features,
            "mask": mask,
            "file_name": file_name,
        }

    def get_map(self, map_name):
        if map_name not in self.maps:
            self.maps[map_name] = read_map(map_name)
        return self.maps[map_name]

    def get_distance_map(self, map_name):
        if map_name not in self.distance_maps:
            self.distance_maps[map_name] = read_distance_map(map_name)
        return self.distance_maps[map_name]

    def process_input_file(self, input_file):
        with open(input_file, "rb") as f:
            agent_num = int(np.frombuffer(f.read(2), dtype=np.uint16)[0])
            all_data = np.frombuffer(f.read(agent_num * 10), dtype=np.uint16)
            agent_locations = all_data[: 2 * agent_num].reshape(agent_num, 2)
            goal_locations = all_data[2 * agent_num : 4 * agent_num].reshape(
                agent_num, 2
            )
            actions = all_data[4 * agent_num : 5 * agent_num].reshape(agent_num)

            return {
                "agent_locations": agent_locations,
                "goal_locations": goal_locations,
                "actions": actions,
            }
