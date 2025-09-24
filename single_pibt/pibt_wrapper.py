"""
单步PIBT Python包装函数
提供简单易用的接口来调用PIBT算法获取下一步动作
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Union

try:
    from . import single_step_pibt_py
except ImportError:
    import single_step_pibt_py

class PIBTSingleStep:
    """单步PIBT求解器"""
    
    def __init__(self, map_width: int, map_height: int, seed: int = 0):
        """
        初始化PIBT求解器
        
        Args:
            map_width: 地图宽度
            map_height: 地图高度  
            seed: 随机种子
        """
        self.solver = single_step_pibt_py.SingleStepPIBT(map_width, map_height, seed)
        self.map_width = map_width
        self.map_height = map_height
    
    def solve_next_actions(self, 
                          current_positions: Union[np.ndarray, torch.Tensor, List[Tuple[int, int]]],
                          goal_positions: Union[np.ndarray, torch.Tensor, List[Tuple[int, int]]],
                          obstacle_map: Union[np.ndarray, torch.Tensor, List[List[int]]],
                          priorities: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
                          action_preferences: Optional[Union[np.ndarray, torch.Tensor, List[List[float]]]] = None,
                          elapsed_times: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None
                          ) -> Union[np.ndarray, List[int]]:
        """
        求解所有agent的下一步动作
        
        Args:
            current_positions: 当前位置，shape: [num_agents, 2] (x, y)
            goal_positions: 目标位置，shape: [num_agents, 2] (x, y)
            obstacle_map: 障碍物地图，shape: [height, width]，0表示可通行，1表示障碍
            priorities: 可选，agent优先级，shape: [num_agents]
            action_preferences: 可选，动作偏好，shape: [num_agents, 5]
            elapsed_times: 可选，已用时间，shape: [num_agents]
            
        Returns:
            动作索引数组，shape: [num_agents] (0=停留, 1=上, 2=下, 3=左, 4=右)
        """
        # 转换输入数据
        current_pos = self._convert_positions(current_positions)
        goal_pos = self._convert_positions(goal_positions)
        obstacle = self._convert_obstacle_map(obstacle_map)
        
        num_agents = len(current_pos)
        
        # 处理优先级
        if priorities is None:
            priorities = [0.0] * num_agents
        else:
            priorities = self._convert_to_list(priorities)
            
        # 处理已用时间
        if elapsed_times is None:
            elapsed_times = [0] * num_agents
        else:
            elapsed_times = self._convert_to_list(elapsed_times, dtype=int)
            
        # 创建Agent状态
        agents = []
        for i in range(num_agents):
            agent = single_step_pibt_py.AgentState(
                id=i,
                current_pos=single_step_pibt_py.Position(current_pos[i][0], current_pos[i][1]),
                goal_pos=single_step_pibt_py.Position(goal_pos[i][0], goal_pos[i][1]),
                priority=priorities[i],
                elapsed_time=elapsed_times[i]
            )
            agents.append(agent)
        
        # 处理动作偏好
        action_prefs = []
        if action_preferences is not None:
            action_prefs = self._convert_action_preferences(action_preferences)
        
        # 调用求解器
        actions = self.solver.get_next_actions(agents, obstacle, action_prefs)
        
        return np.array(actions)
    
    def _convert_positions(self, positions) -> List[Tuple[int, int]]:
        """转换位置数据为列表格式"""
        if torch.is_tensor(positions):
            positions = positions.cpu().numpy()
        elif isinstance(positions, np.ndarray):
            pass
        elif isinstance(positions, list):
            positions = np.array(positions)
        else:
            raise ValueError(f"不支持的位置数据类型: {type(positions)}")
            
        return [(int(pos[0]), int(pos[1])) for pos in positions]
    
    def _convert_obstacle_map(self, obstacle_map) -> List[List[int]]:
        """转换障碍物地图"""
        if torch.is_tensor(obstacle_map):
            obstacle_map = obstacle_map.cpu().numpy()
        elif isinstance(obstacle_map, np.ndarray):
            pass
        elif isinstance(obstacle_map, list):
            return obstacle_map
        else:
            raise ValueError(f"不支持的障碍物地图类型: {type(obstacle_map)}")
            
        return [[int(cell) for cell in row] for row in obstacle_map]
    
    def _convert_to_list(self, data, dtype=float):
        """转换数据为列表"""
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            pass
        elif isinstance(data, list):
            return [dtype(x) for x in data]
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
            
        return [dtype(x) for x in data]
    
    def _convert_action_preferences(self, action_preferences) -> List[List[float]]:
        """转换动作偏好，从概率分布中采样动作作为偏好"""
        if torch.is_tensor(action_preferences):
            action_preferences = action_preferences.cpu().numpy()
        elif isinstance(action_preferences, np.ndarray):
            pass
        elif isinstance(action_preferences, list):
            action_preferences = np.array(action_preferences)
        else:
            raise ValueError(f"不支持的动作偏好类型: {type(action_preferences)}")

        # 从每个agent的动作概率分布中采样5个动作作为偏好序列
        preferences = []
        for agent_probs in action_preferences:
            # 确保概率和为1
            agent_probs = agent_probs / (np.sum(agent_probs) + 1e-8)

            # 从概率分布中采样5个动作作为偏好序列
            agent_preferences = np.random.choice(5, size=5, p=agent_probs, replace=True).tolist()
            preferences.append(agent_preferences)

        return preferences


# 全局求解器缓存
_solver_cache = {}

def clear_pibt_cache():
    """清理PIBT求解器缓存"""
    global _solver_cache
    _solver_cache.clear()

def pibt_solve_single_step(env,
                          action_probabilities: Union[np.ndarray, torch.Tensor],
                          priorities: Optional[Union[np.ndarray, torch.Tensor]] = None,
                          seed: int = 0) -> Union[np.ndarray, torch.Tensor]:
    """
    便捷函数：使用PIBT求解单步动作
    专门为MAPF环境设计的简单接口

    Args:
        env: MAPF环境对象
        action_probabilities: 动作概率，shape: [num_agents, 5]
        priorities: 可选的优先级，shape: [num_agents]
        seed: 随机种子

    Returns:
        动作索引，shape: [num_agents]
    """
    # 复用求解器实例（基于地图尺寸作为key）
    cache_key = (env.width, env.height, seed)
    if cache_key not in _solver_cache:
        _solver_cache[cache_key] = PIBTSingleStep(env.width, env.height, seed)

    solver = _solver_cache[cache_key]
    
    # 获取当前和目标位置
    current_positions = env.agent_positions.cpu().numpy()
    goal_positions = env.goal_positions.cpu().numpy()
    
    # 获取障碍物地图
    if hasattr(env, 'obstacle_map'):
        obstacle_map = env.obstacle_map
    elif hasattr(env, 'map_data'):
        obstacle_map = env.map_data
    else:
        # 如果没有障碍物地图，创建一个全0的地图
        obstacle_map = np.zeros((env.height, env.width), dtype=int)
    
    # 生成随机优先级（如果没有提供）
    if priorities is None:
        np.random.seed(seed)
        priorities = np.random.rand(env.num_agents)
    
    # 求解
    actions = solver.solve_next_actions(
        current_positions=current_positions,
        goal_positions=goal_positions,
        obstacle_map=obstacle_map,
        priorities=priorities,
        action_preferences=action_probabilities
    )
    
    # 返回与输入相同的数据类型
    if torch.is_tensor(action_probabilities):
        return torch.from_numpy(actions).to(action_probabilities.device)
    else:
        return actions
