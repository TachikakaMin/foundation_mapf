"""
简化的MAPF强化学习环境
- 随机生成test case
- 使用项目的特征构建
- 接收神经网络输出
- 无error handling，让程序crash
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
from tools.utils import construct_input_feature, create_distance_map
from tools.path_formation import move_agent


class MAPFEnv:
    def __init__(self, 
                 height: int = 32,
                 width: int = 32,
                 num_agents: int = 16,
                 obstacle_density: float = 0.2,
                 max_steps: int = 200,
                 feature_dim: int = 6,
                 feature_type: str = "gradient",
                 map_data: np.ndarray = None,
                 distance_map = None):
        self.height = height
        self.width = width
        self.num_agents = num_agents
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        self.priorities = torch.rand((num_agents,),dtype=torch.float32)
        # 环境状态
        self.current_step = 0
        self.map_data = None
        self.distance_map = None
        self.agent_positions = None
        self.goal_positions = None
        self.temperature = None
        
        # 动作映射
        self.action_deltas = {
            0: (0, 0),   # stay
            1: (0, 1),   # up  
            2: (0, -1),  # down
            3: (-1, 0),  # left
            4: (1, 0),   # right
        }
        
        # 使用预先提供的地图或生成新地图
        if map_data is not None:
            self.map_data = map_data
        else:
            self.map_data = self._generate_map()
        
        # 使用预先计算的距离地图或生成新的
        if distance_map is not None:
            self.distance_map = distance_map
        else:
            self.distance_map = create_distance_map(self.map_data)        
        
    def _generate_map(self):
        """随机生成地图"""
        map_data = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 随机放置障碍物
        num_obstacles = int(self.height * self.width * self.obstacle_density)
        obstacle_positions = random.sample(
            [(i, j) for i in range(self.height) for j in range(self.width)],
            num_obstacles
        )
        
        for x, y in obstacle_positions:
            map_data[x, y] = 1.0
            
        return map_data
    
    def _get_free_positions(self):
        """获取可通行位置"""
        free_positions = []
        for i in range(self.height):
            for j in range(self.width):
                if self.map_data[i, j] == 0:
                    free_positions.append((i, j))
        return free_positions
    
    def _generate_agents_and_goals(self):
        """随机生成agent起始位置和目标位置"""
        free_positions = self._get_free_positions()
        agent_positions = random.sample(free_positions, self.num_agents)
        goal_positions = random.sample(free_positions, self.num_agents)
        return agent_positions, goal_positions
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.reward_for_each_agent = np.ones(self.num_agents, dtype=int)
        # 生成agent和goal位置
        self.agent_positions, self.goal_positions = self._generate_agents_and_goals()
        
        # 转换为tensor
        self.agent_positions = torch.tensor(self.agent_positions, dtype=torch.long)
        self.goal_positions = torch.tensor(self.goal_positions, dtype=torch.long)
        
        # 初始化温度
        self.temperature = torch.ones(self.num_agents, dtype=torch.float32)
        
        # 构建观测
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self):
        """构建观测特征"""
        input_features = construct_input_feature(
            self.map_data,
            self.agent_positions,
            self.goal_positions,
            self.distance_map,
            self.feature_dim,
            self.feature_type
        )
        return input_features.numpy()
    
    def get_feature(self):
        """获取当前特征tensor（用于action采样）"""
        input_features = construct_input_feature(
            self.map_data,
            self.agent_positions,
            self.goal_positions,
            self.distance_map,
            self.feature_dim,
            self.feature_type
        )
        return input_features
    

    
    def step(self, action_map):
        """执行一步
        
        Args:
            action_map: 已采样的动作地图 [height, width]
        """
        self.current_step += 1
        
        # 转换numpy数组为torch张量（如果需要）
        if isinstance(action_map, np.ndarray):
            action_map = torch.from_numpy(action_map)
        
        # 移动agent
        new_positions, new_temperature = move_agent(
            action_map,
            torch.from_numpy(self.map_data),
            self.agent_positions,
            self.temperature
        )
        
        
        self.agent_positions = new_positions
        self.temperature = new_temperature
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查终止条件
        done = self._check_done()
        truncated = self.current_step >= self.max_steps
        
        # 构建新观测和信息
        next_obs = self._get_observation()
        info = self._get_info()
        
        return next_obs, reward, done, truncated, info
    
    def _calculate_reward(self):
        """计算奖励"""
        reward = 0.0
        
        # 时间惩罚
        reward -= 1.0/self.max_steps
        
        # 到达目标的奖励
        for i in range(self.num_agents):
            if torch.equal(self.agent_positions[i], self.goal_positions[i]):
                reward += self.reward_for_each_agent[i]/self.num_agents
                self.reward_for_each_agent[i] = 0
            elif self.reward_for_each_agent[i] == 0:
                self.reward_for_each_agent[i] = 1
                reward -= self.reward_for_each_agent[i]/self.num_agents
                
        if self.reward_for_each_agent.sum() == 0:
            reward += 2
            
        return reward
    
    def _check_done(self):
        """检查是否完成"""
        return self.reward_for_each_agent.sum() == 0
    
    def _get_info(self):
        """获取环境信息"""
        success_count = sum(
            1 for i in range(self.num_agents) 
            if torch.equal(self.agent_positions[i], self.goal_positions[i])
        )
        
        # 构建agent mask
        agent_mask = torch.zeros((self.height, self.width), dtype=torch.bool)
        for i in range(self.num_agents):
            x, y = self.agent_positions[i].tolist()
            agent_mask[x, y] = True
        
        return {
            'agent_positions': self.agent_positions.clone(),
            'goal_positions': self.goal_positions.clone(),
            'current_step': self.current_step,
            'success_count': success_count,
            'success_rate': success_count / self.num_agents,
            'agent_mask': agent_mask,
            'map_shape': (self.height, self.width),
        }
    
    def render(self):
        """渲染环境（文本版）"""
        print(f"\nStep {self.current_step}")
        print(f"Success: {self._get_info()['success_count']}/{self.num_agents}")
        
        # 创建渲染地图
        render_map = self.map_data.copy().astype(int)
        
        # 标记目标
        for i in range(self.num_agents):
            x, y = self.goal_positions[i].tolist()
            if render_map[x, y] == 0:
                render_map[x, y] = 3  # 目标标记
        
        # 标记agent（覆盖目标）
        for i in range(self.num_agents):
            x, y = self.agent_positions[i].tolist()
            render_map[x, y] = 2  # agent标记
        
        # 打印地图的一部分
        display_size = min(20, self.height, self.width)
        symbols = {0: '.', 1: '#', 2: 'A', 3: 'G'}
        
        for i in range(display_size):
            row = ""
            for j in range(display_size):
                row += symbols.get(render_map[i, j], '?')
            print(row)
        
        if self.height > display_size or self.width > display_size:
            print(f"... (显示 {display_size}x{display_size} / {self.height}x{self.width})")
    
    def render_frame(self, dpi=80, figsize=(8, 8)):
        """渲染环境为图像帧 - 参考tools中的专业可视化风格"""
        # 创建图像
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 绘制地图（使用binary colormap，障碍物黑色，空地白色）
        ax.imshow(self.map_data, cmap="binary")
        
        # 设置专业网格
        ax.grid(True, which="major", color="gray", linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        ax.set_xticks(np.arange(0, self.width, 1), minor=True)
        ax.set_yticks(np.arange(0, self.height, 1), minor=True)
        ax.tick_params(which="minor", length=0)
        
        # 设置网格标签
        ax.set_xticklabels([], minor=False)
        ax.set_yticklabels([], minor=False)
        ax.set_xticklabels(range(self.width), minor=True, fontsize=8)
        ax.set_yticklabels(range(self.height), minor=True, fontsize=8)
        
        # 使用专业配色方案
        colors = ["r", "g", "b", "y", "m", "c", "orange", "purple"]
        
        # 绘制目标位置（空心星形）
        for i in range(self.num_agents):
            x, y = self.goal_positions[i].tolist()
            color = colors[i % len(colors)]
            ax.scatter(y, x, 
                      edgecolor=color,
                      facecolor="none", 
                      marker="*", 
                      s=200,
                      linewidth=2)
            
            # 添加目标ID标注
            ax.annotate(str(i),
                       xy=(y, x),
                       xytext=(0, 0),
                       color="black",
                       ha="center", va="center",
                       fontsize=10, fontweight='bold')
        
        # 绘制agent位置（实心圆）
        for i in range(self.num_agents):
            x, y = self.agent_positions[i].tolist()
            color = colors[i % len(colors)]
            ax.scatter(y, x, 
                      c=color, 
                      s=100, 
                      edgecolor='black',
                      linewidth=2)
            
            # 添加agent ID标注（白色背景圆形）
            ax.annotate(str(i),
                       xy=(y, x),
                       xytext=(0, 0),
                       bbox=dict(boxstyle="circle", fc="white", ec=color, linewidth=1),
                       ha="center", va="center",
                       fontsize=8, fontweight='bold')
        
        # 设置坐标轴
        ax.set_xlim(-0.5, self.width-0.5)
        ax.set_ylim(-0.5, self.height-0.5)
        ax.set_aspect('equal')
        
        # 添加专业标题
        info = self._get_info()
        ax.set_title(f"MAPF Step {self.current_step}: {info['success_count']}/{self.num_agents} agents reached goals ({info['success_rate']:.1%})", 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例说明
        ax.text(0.02, 0.98, f"● Agents  ★ Goals", 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')
        
        # 转换为numpy数组
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        h, w = fig.canvas.get_width_height()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        
        plt.close(fig)
        return frame


# 测试代码
if __name__ == "__main__":
    print("测试简化MAPF环境...")
    
    # 创建环境
    env = MAPFEnv(height=16, width=16, num_agents=4, max_steps=50)
    
    # 重置环境
    obs, info = env.reset()
    print(f"观测形状: {obs.shape}")
    print(f"代理数量: {info['success_count']}/{env.num_agents}")
    
    # 模拟几个步骤
    for step in range(5):
        # 随机网络输出
        network_output = torch.randn(5, env.height, env.width)
        
        # 从logits中采样动作
        from tools.path_formation import sample_action
        action_map = sample_action(
            network_output.unsqueeze(0),  # 添加batch维度
            env.agent_positions, 
            env.temperature,
            env.get_feature(),
            action_choice="sample"
        )
        
        next_obs, reward, done, truncated, info = env.step(action_map)
        
        print(f"\n步骤 {step + 1}:")
        print(f"奖励: {reward:.2f}")
        print(f"完成: {done}")
        print(f"成功率: {info['success_rate']:.2%}")
        
        if done:
            print("任务完成!")
            break
    
    # 渲染最终状态
    env.render()
    print("\n测试完成!") 