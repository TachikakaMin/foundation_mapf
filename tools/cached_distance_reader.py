"""
带缓存的距离地图读取器

避免重复加载相同的距离地图文件
"""

import os
from .distance_map_reader import DistanceMapReader
import pickle
from functools import lru_cache

# 全局缓存
_distance_map_cache = {}

class CachedDistanceMapReader:
    """带缓存的距离地图读取器"""
    
    @staticmethod
    def get_distance_map(map_file_path):
        """获取距离地图，使用缓存避免重复加载"""
        global _distance_map_cache
        
        # 首先尝试C++生成的二进制格式
        cpp_file_path = map_file_path.replace("map_files", "distance_maps").replace(".map", ".dmap")
        
        if cpp_file_path in _distance_map_cache:
            return _distance_map_cache[cpp_file_path]
        
        if os.path.exists(cpp_file_path):
            try:
                reader = DistanceMapReader(cpp_file_path)
                _distance_map_cache[cpp_file_path] = reader
                return reader
            except Exception:
                pass  # 如果加载失败，回退到pickle格式
        
        # 回退到Python的pickle格式
        pkl_file_path = map_file_path.replace("map_files", "distance_maps").replace(".map", ".pkl")
        
        if pkl_file_path in _distance_map_cache:
            return _distance_map_cache[pkl_file_path]
        
        if os.path.exists(pkl_file_path):
            with open(pkl_file_path, "rb") as f:
                distance_map = pickle.load(f)
                _distance_map_cache[pkl_file_path] = distance_map
                return distance_map
        
        # 如果两种格式都不存在，抛出错误
        raise FileNotFoundError(f"距离地图文件不存在: {cpp_file_path} 或 {pkl_file_path}")

def read_distance_map_cached(map_file_path):
    """
    缓存版本的距离地图读取函数
    """
    return CachedDistanceMapReader.get_distance_map(map_file_path)

def clear_distance_map_cache():
    """清除距离地图缓存"""
    global _distance_map_cache
    _distance_map_cache.clear()
    print("距离地图缓存已清除")

def get_cache_info():
    """获取缓存信息"""
    global _distance_map_cache
    return {
        'cached_maps': len(_distance_map_cache),
        'cache_keys': list(_distance_map_cache.keys())
    } 