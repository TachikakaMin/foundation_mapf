"""
距离地图读取器

用于读取C++版本生成的二进制距离地图文件(.dmap)
"""

import struct
import numpy as np
from typing import Dict, Tuple
import os

NOT_FOUND_PATH = 2048

class DistanceMapReader:
    """二进制距离地图读取器"""
    
    def __init__(self, dmap_file: str):
        """
        初始化距离地图读取器
        
        Args:
            dmap_file: .dmap文件路径
        """
        self.dmap_file = dmap_file
        self.distance_map = {}
        self.height = 0
        self.width = 0
        self._load_distance_map()
    
    def _load_distance_map(self):
        """从二进制文件加载距离地图"""
        if not os.path.exists(self.dmap_file):
            raise FileNotFoundError(f"距离地图文件不存在: {self.dmap_file}")
        
        with open(self.dmap_file, 'rb') as f:
            # 读取文件头部
            num_positions = struct.unpack('I', f.read(4))[0]
            self.height = struct.unpack('I', f.read(4))[0]
            self.width = struct.unpack('I', f.read(4))[0]
            
            # 读取每个位置的距离数据
            for _ in range(num_positions):
                # 读取起始位置
                start_x = struct.unpack('H', f.read(2))[0]
                start_y = struct.unpack('H', f.read(2))[0]
                
                # 读取距离矩阵
                distances = np.zeros((self.height, self.width), dtype=np.int32)
                for x in range(self.height):
                    for y in range(self.width):
                        distance = struct.unpack('H', f.read(2))[0]
                        distances[x][y] = distance if distance < 65535 else NOT_FOUND_PATH
                
                self.distance_map[(start_x, start_y)] = distances
    
    def get_distance(self, agent_location: Tuple[int, int], goal_location: Tuple[int, int]) -> int:
        """
        获取从智能体位置到目标位置的距离
        
        Args:
            agent_location: 智能体位置 (x, y)
            goal_location: 目标位置 (x, y)
            
        Returns:
            距离值, 如果无法到达则返回NOT_FOUND_PATH
        """
        agent_location = (int(agent_location[0]), int(agent_location[1]))
        goal_location = (int(goal_location[0]), int(goal_location[1]))
        
        if agent_location not in self.distance_map:
            return NOT_FOUND_PATH
        
        goal_x, goal_y = goal_location
        if 0 <= goal_x < self.height and 0 <= goal_y < self.width:
            return int(self.distance_map[agent_location][goal_x][goal_y])
        else:
            return NOT_FOUND_PATH
    
    def get_all_distances_from(self, start_location: Tuple[int, int]) -> np.ndarray:
        """
        获取从指定位置到所有位置的距离矩阵
        
        Args:
            start_location: 起始位置 (x, y)
            
        Returns:
            距离矩阵 (height x width)
        """
        start_location = (int(start_location[0]), int(start_location[1]))
        
        if start_location not in self.distance_map:
            # 返回全为NOT_FOUND_PATH的矩阵
            return np.full((self.height, self.width), NOT_FOUND_PATH, dtype=np.int32)
        
        return self.distance_map[start_location].copy()
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        检查位置是否有效（即是否在距离地图中）
        
        Args:
            position: 位置 (x, y)
            
        Returns:
            True如果位置有效, False否则
        """
        position = (int(position[0]), int(position[1]))
        return position in self.distance_map
    
    def get_valid_positions(self):
        """
        获取所有有效位置
        
        Returns:
            有效位置列表
        """
        return list(self.distance_map.keys())
    
    def get_map_size(self) -> Tuple[int, int]:
        """
        获取地图尺寸
        
        Returns:
            (height, width)
        """
        return self.height, self.width


def read_distance_map_cpp(map_file_path: str) -> DistanceMapReader:
    """
    读取C++生成的距离地图文件
    
    Args:
        map_file_path: 地图文件路径 (.map)
        
    Returns:
        DistanceMapReader对象
    """
    # 将.map路径转换为.dmap路径
    dmap_path = map_file_path.replace("map_files", "distance_maps").replace(".map", ".dmap")
    
    if not os.path.exists(dmap_path):
        raise FileNotFoundError(f"距离地图文件不存在: {dmap_path}")
    
    return DistanceMapReader(dmap_path)


def get_distance_cpp(distance_map: DistanceMapReader, agent_location, goal_location) -> int:
    """
    获取距离的便捷函数, 兼容原有的接口
    
    Args:
        distance_map: DistanceMapReader对象
        agent_location: 智能体位置
        goal_location: 目标位置
        
    Returns:
        距离值
    """
    return distance_map.get_distance(agent_location, goal_location)


# 为了兼容性, 提供与原有Python版本相同的接口
def create_distance_map_dict_from_cpp(dmap_file: str) -> Dict:
    """
    从C++生成的距离地图文件创建Python字典格式的距离地图
    
    Args:
        dmap_file: .dmap文件路径
        
    Returns:
        Python字典格式的距离地图
    """
    reader = DistanceMapReader(dmap_file)
    return reader.distance_map


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python distance_map_reader.py <dmap文件路径>")
        sys.exit(1)
    
    dmap_file = sys.argv[1]
    
    reader = DistanceMapReader(dmap_file)
    print(f"成功加载距离地图: {dmap_file}")
    print(f"地图尺寸: {reader.get_map_size()}")
    print(f"有效位置数量: {len(reader.get_valid_positions())}")
    
    # 测试一些距离查询
    valid_positions = reader.get_valid_positions()
    if len(valid_positions) >= 2:
        pos1 = valid_positions[0]
        pos2 = valid_positions[1]
        distance = reader.get_distance(pos1, pos2)
        print(f"从 {pos1} 到 {pos2} 的距离: {distance}")