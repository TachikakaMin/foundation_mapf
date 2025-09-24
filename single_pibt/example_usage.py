"""
单步PIBT使用示例
展示如何使用新的PIBT包装来获取下一步动作
"""

import numpy as np
import torch
from pibt_wrapper import PIBTSingleStep, pibt_solve_single_step

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建求解器 (地图大小: 5x5)
    solver = PIBTSingleStep(map_width=5, map_height=5, seed=42)
    
    # 定义agent状态
    current_positions = [(1, 1), (3, 3)]  # 2个agent
    goal_positions = [(3, 3), (1, 1)]     # 交换位置
    
    # 创建障碍物地图 (0=可通行, 1=障碍物)
    obstacle_map = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],  # 中间有个障碍物
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    
    # 定义动作偏好 (每个agent对5个动作的偏好)
    # 动作顺序: 0=停留, 1=上, 2=下, 3=左, 4=右
    action_preferences = [
        [0.1, 0.2, 0.3, 0.2, 0.2],  # agent 0偏好向下
        [0.1, 0.3, 0.2, 0.2, 0.2]   # agent 1偏好向上
    ]
    
    # 求解下一步动作
    actions = solver.solve_next_actions(
        current_positions=current_positions,
        goal_positions=goal_positions,
        obstacle_map=obstacle_map,
        action_preferences=action_preferences
    )
    
    print(f"当前位置: {current_positions}")
    print(f"目标位置: {goal_positions}")
    print(f"下一步动作: {actions}")
    
    # 解释动作
    action_names = ["停留", "上", "下", "左", "右"]
    for i, action in enumerate(actions):
        print(f"Agent {i}: {action_names[action]}")

def example_with_torch():
    """使用PyTorch张量的示例"""
    print("\n=== PyTorch张量示例 ===")
    
    # 创建求解器
    solver = PIBTSingleStep(map_width=8, map_height=8, seed=0)
    
    # 使用PyTorch张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_positions = torch.tensor([[0, 0], [7, 7], [0, 7], [7, 0]], device=device)
    goal_positions = torch.tensor([[7, 7], [0, 0], [7, 0], [0, 7]], device=device)
    
    # 创建复杂一点的地图
    obstacle_map = torch.zeros((8, 8), dtype=torch.int32, device=device)
    obstacle_map[3:5, 3:5] = 1  # 中间区域有障碍物
    
    # 随机动作偏好
    torch.manual_seed(42)
    action_preferences = torch.softmax(torch.randn(4, 5), dim=-1)
    
    actions = solver.solve_next_actions(
        current_positions=current_positions,
        goal_positions=goal_positions,
        obstacle_map=obstacle_map,
        action_preferences=action_preferences
    )
    
    print(f"设备: {device}")
    print(f"当前位置:\n{current_positions}")
    print(f"目标位置:\n{goal_positions}")
    print(f"下一步动作: {actions}")

def example_environment_integration():
    """与MAPF环境集成的示例"""
    print("\n=== 环境集成示例 ===")
    
    # 模拟环境对象
    class MockEnv:
        def __init__(self):
            self.width = 10
            self.height = 10
            self.num_agents = 6
            self.agent_positions = torch.tensor([
                [1, 1], [8, 8], [1, 8], [8, 1], [5, 5], [2, 7]
            ])
            self.goal_positions = torch.tensor([
                [8, 8], [1, 1], [8, 1], [1, 8], [2, 2], [7, 2]
            ])
            # 创建有障碍物的地图
            self.map_data = np.zeros((10, 10), dtype=int)
            self.map_data[4:6, 4:6] = 1  # 中心区域障碍物
    
    # 创建模拟环境
    env = MockEnv()
    
    # 模拟神经网络输出的动作概率
    np.random.seed(123)
    action_probabilities = np.random.rand(env.num_agents, 5)
    # 归一化为概率
    action_probabilities = action_probabilities / action_probabilities.sum(axis=1, keepdims=True)
    
    # 使用便捷函数求解
    actions = pibt_solve_single_step(env, action_probabilities, seed=42)
    
    print(f"环境大小: {env.width}x{env.height}")
    print(f"Agent数量: {env.num_agents}")
    print(f"当前位置:\n{env.agent_positions}")
    print(f"目标位置:\n{env.goal_positions}")
    print(f"动作概率形状: {action_probabilities.shape}")
    print(f"PIBT求解的动作: {actions}")

if __name__ == "__main__":
    try:
        example_basic_usage()
        example_with_torch()
        example_environment_integration()
        print("\n✅ 所有示例运行成功！")
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请先编译模块: ./build.sh")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
