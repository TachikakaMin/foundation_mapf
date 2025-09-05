"""
MAPF工具集

重新组织的工具集，包含：
- core: 核心工具函数
- data_processing: 数据处理和预计算
- extensions: C++扩展
- benchmarks: 性能测试
- converters: 数据转换器
- visualization: 可视化工具
- testing: 测试工具
"""

# 为了向后兼容，保持原有的导入接口
from .core.utils import *
from .core.path_formation import *
from .data_processing.cached_distance_reader import read_distance_map_cached
from .data_processing.distance_map_reader import DistanceMapReader

__version__ = "2.0.0"
