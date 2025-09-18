import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from multiprocessing import Pool
import tqdm


def parse_coordinates(coord_str):
    """Parse LACAM coordinates"""
    # 使用正则表达式提取所有坐标对
    coord_pairs = re.findall(r"\((\d+),(\d+)\)", coord_str)
    
    # 解析每个坐标对, 并交换 x 和 y
    coords = [(int(y), int(x)) for x, y in coord_pairs]
    
    return coords


def read_map_file(map_path):
    """读取map文件，返回障碍物地图"""
    with open(map_path, 'r') as f:
        lines = f.readlines()
    
    # 解析地图文件
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    
    map_data = np.zeros((height, width), dtype=np.float32)
    
    # 从第4行开始是地图数据
    for i, line in enumerate(lines[4:4+height]):
        for j, char in enumerate(line.strip()):
            if char == '@' or char == 'T':  # 障碍物
                map_data[i][j] = 1.0
    
    return map_data


def parse_path_file(path_file):
    """解析path文件，返回starts, goals, paths"""
    with open(path_file, 'r') as f:
        lines = f.readlines()
    
    # 解析起始位置和目标位置
    starts_line = None
    goals_line = None
    solution_line_idx = -1
    
    for i, line in enumerate(lines):
        if line.startswith('starts='):
            starts_line = line.strip()[7:]  # 去掉'starts='
        elif line.startswith('goals='):
            goals_line = line.strip()[6:]   # 去掉'goals='
        elif line.startswith('solution='):
            solution_line_idx = i
            break
    
    if not starts_line or not goals_line or solution_line_idx == -1:
        return None
    
    # 解析起始位置和目标位置
    starts = parse_coordinates(starts_line)
    goals = parse_coordinates(goals_line)
    
    # 解析路径数据
    paths = []
    for line in lines[solution_line_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        # 格式: "0:(x1,y1),(x2,y2),..."
        colon_pos = line.find(':')
        if colon_pos != -1:
            coord_str = line[colon_pos + 1:]
            coords = parse_coordinates(coord_str)
            if coords:
                paths.append(coords)
    
    return starts, goals, paths


# 移除单智能体模式，只保留多智能体模式

class MultiAgentPathPlanningDataset(Dataset):
    """用于扩散模型路径规划的数据集（多智能体，无碰撞）- 从path文件读取"""
    
    def __init__(
        self,
        data_root="data",
        grid_size=(32,32),
        num_agents=3,
        num_samples=None,
        max_timesteps=32,
        pattern="*"
    ):
        self.data_root = data_root
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_samples = num_samples
        self.max_timesteps = max_timesteps
        
        # 收集指定智能体数量的path文件
        path_files = self._collect_path_files(pattern, num_agents)
        
        # 加载数据
        self.data = self._load_data(path_files)
        
        if num_samples and len(self.data) > num_samples:
            self.data = self.data[:num_samples]
        
        print(f"MultiAgentPathPlanningDataset loaded: {len(self.data)} episodes")

    def _collect_path_files(self, pattern, num_agents):
        """收集指定智能体数量的path文件"""
        path_files_dir = os.path.join(self.data_root, "path_files")
        
        if not os.path.exists(path_files_dir):
            print(f"Path files directory does not exist: {path_files_dir}")
            return []
        
        # 直接用glob匹配包含指定智能体数量的文件
        # 文件名格式: maze-*-*-*-*-*-*-{num_agents}-*.path
        pattern = f"**/*-{num_agents}-*.path"
        path_files = glob.glob(os.path.join(path_files_dir, pattern), recursive=True)
        
        print(f"Found {len(path_files)} path files matching {num_agents} agents")
        return path_files
    
    def _check_agent_count(self, path_file, expected_agents):
        """检查path文件的智能体数量（从文件名解析）"""
        try:
            filename = os.path.basename(path_file)
            # 文件名格式: maze-32-32-30-3-75-0-16-1.path
            parts = filename.replace('.path', '').split('-')
            if len(parts) >= 8:
                # 智能体数量是倒数第二个部分
                agent_count = int(parts[-2])
                return agent_count == expected_agents
        except:
            pass
        return False
    
    def _load_data(self, path_files):
        """加载数据"""
        data = []
        
        # 如果指定了样本数量限制，设置目标数量
        target_samples = self.num_samples if self.num_samples else len(path_files)
        
        pbar = tqdm.tqdm(total=target_samples, desc="Loading multi-agent path files")
        
        for path_file in path_files:
            # 如果已经达到目标样本数，停止加载
            if len(data) >= target_samples:
                break
            
            # 更新进度条描述显示当前文件
            filename = os.path.basename(path_file)
            pbar.set_description(f"Loading MA: {filename}")
                
            try:
                # 解析path文件
                result = parse_path_file(path_file)
                if result is None:
                    continue
                
                starts, goals, paths = result
                
                # 检查智能体数量
                if len(starts) != self.num_agents:
                    continue
                
                # 获取对应的map文件
                map_file = self._get_map_file(path_file)
                if not map_file or not os.path.exists(map_file):
                    continue
                
                # 读取地图
                obstacle_map = read_map_file(map_file)
                
                # 检查地图尺寸
                if obstacle_map.shape != self.grid_size:
                    continue
                
                # 填充路径到统一长度
                max_len = min(len(paths), self.max_timesteps)
                padded_paths = []
                for agent_id in range(self.num_agents):
                    agent_path = []
                    for t in range(max_len):
                        if t < len(paths) and agent_id < len(paths[t]):
                            agent_path.append(paths[t][agent_id])
                        else:
                            # 使用目标位置填充
                            agent_path.append(goals[agent_id])
                    padded_paths.append(agent_path)
                
                data.append({
                    'obstacle_map': obstacle_map,
                    'starts': starts,
                    'goals': goals,
                    'paths': padded_paths,
                    'path_length': max_len
                })
                
                # 更新进度条
                pbar.update(1)
                
            except Exception as e:
                continue
        
        pbar.close()
        return data
    
    def _get_map_file(self, path_file):
        """根据path文件路径获取对应的map文件路径"""
        filename = os.path.basename(path_file)
        parts = filename.replace('.path', '').split('-')
        
        if len(parts) >= 7:
            map_name = '-'.join(parts[:6]) + f"-{parts[6]}.map"
            map_subdir = '-'.join(parts[:6])
            map_path = os.path.join(self.data_root, "map_files", map_subdir, map_name)
            return map_path
        
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        obstacle_map = torch.tensor(item['obstacle_map'], dtype=torch.float32).unsqueeze(0)
        
        # Create path map without time dimension for 2D version
        path_map = torch.zeros((1, self.grid_size[0], self.grid_size[1]), dtype=torch.float32)
        
        # Fill in the path map - use different values for different agents
        for agent_id, agent_path in enumerate(item['paths']):
            agent_value = float(agent_id + 1) / float(self.num_agents)
            for r, c in agent_path:
                if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                    path_map[0, r, c] = agent_value
        
        starts = torch.tensor(item['starts'])
        goals = torch.tensor(item['goals'])
        
        return obstacle_map, path_map, starts, goals


class MultiAgent3DPathPlanningDataset(Dataset):
    """用于扩散模型路径规划的数据集（多智能体3D，无碰撞）- 从path文件读取"""
    
    def __init__(
        self,
        data_root="data",
        grid_size=(32,32),
        num_agents=3,
        num_samples=None,
        max_timesteps=32,
        pattern="*"
    ):
        self.data_root = data_root
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_samples = num_samples
        self.max_timesteps = max_timesteps
        
        # 收集指定智能体数量的path文件
        path_files = self._collect_path_files(pattern, num_agents)
        
        # 加载数据
        self.data = self._load_data(path_files)
        
        if num_samples and len(self.data) > num_samples:
            self.data = self.data[:num_samples]
        
        print(f"3D Multi-agent dataset generated: {len(self.data)} episodes")

    def _collect_path_files(self, pattern, num_agents):
        """收集指定智能体数量的path文件"""
        path_files_dir = os.path.join(self.data_root, "path_files")
        
        if not os.path.exists(path_files_dir):
            print(f"Path files directory does not exist: {path_files_dir}")
            return []
        
        # 直接用glob匹配包含指定智能体数量的文件
        # 文件名格式: maze-*-*-*-*-*-*-{num_agents}-*.path
        pattern = f"**/*-{num_agents}-*.path"
        path_files = glob.glob(os.path.join(path_files_dir, pattern), recursive=True)
        
        print(f"Found {len(path_files)} path files matching {num_agents} agents")
        return path_files
    
    def _check_agent_count(self, path_file, expected_agents):
        """检查path文件的智能体数量（从文件名解析）"""
        try:
            filename = os.path.basename(path_file)
            # 文件名格式: maze-32-32-30-3-75-0-16-1.path
            parts = filename.replace('.path', '').split('-')
            if len(parts) >= 8:
                # 智能体数量是倒数第二个部分
                agent_count = int(parts[-2])
                return agent_count == expected_agents
        except:
            pass
        return False
    
    def _load_data(self, path_files):
        """加载数据"""
        data = []
        
        # 如果指定了样本数量限制，设置目标数量
        target_samples = self.num_samples if self.num_samples else len(path_files)
        
        pbar = tqdm.tqdm(total=target_samples, desc="Loading 3D multi-agent path files")
        
        for path_file in path_files:
            # 如果已经达到目标样本数，停止加载
            if len(data) >= target_samples:
                break
            
            # 更新进度条描述显示当前文件
            filename = os.path.basename(path_file)
            pbar.set_description(f"Loading 3D: {filename}")
                
            try:
                # 解析path文件
                result = parse_path_file(path_file)
                if result is None:
                    continue
                
                starts, goals, paths = result
                
                # 检查智能体数量
                if len(starts) != self.num_agents:
                    continue
                
                # 获取对应的map文件
                map_file = self._get_map_file(path_file)
                if not map_file or not os.path.exists(map_file):
                    continue
                
                # 读取地图
                obstacle_map = read_map_file(map_file)
                
                # 检查地图尺寸
                if obstacle_map.shape != self.grid_size:
                    continue
                
                # 填充路径到统一长度，按智能体组织
                max_len = min(len(paths), self.max_timesteps)
                padded_paths = []
                for agent_id in range(self.num_agents):
                    agent_path = []
                    for t in range(max_len):
                        if t < len(paths) and agent_id < len(paths[t]):
                            agent_path.append(paths[t][agent_id])
                        else:
                            # 使用目标位置填充
                            agent_path.append(goals[agent_id])
                    
                    # 如果路径长度小于max_timesteps，用目标位置填充剩余时间步
                    while len(agent_path) < self.max_timesteps:
                        agent_path.append(goals[agent_id])
                    
                    padded_paths.append(agent_path)
                
                data.append({
                    'obstacle_map': obstacle_map,
                    'starts': starts,
                    'goals': goals,
                    'paths': padded_paths,  # [num_agents, max_timesteps, 2]
                    'path_length': max_len
                })
                
                # 更新进度条
                pbar.update(1)
                
            except Exception as e:
                continue
        
        pbar.close()
        return data
    
    def _get_map_file(self, path_file):
        """根据path文件路径获取对应的map文件路径"""
        filename = os.path.basename(path_file)
        parts = filename.replace('.path', '').split('-')
        
        if len(parts) >= 7:
            map_name = '-'.join(parts[:6]) + f"-{parts[6]}.map"
            map_subdir = '-'.join(parts[:6])
            map_path = os.path.join(self.data_root, "map_files", map_subdir, map_name)
            return map_path
        
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Convert obstacle map to tensor with shape [1, T, W, H]
        obstacle_map = torch.tensor(item['obstacle_map'], dtype=torch.float32).unsqueeze(0)
        
        # Create 3D path map with time dimension [1, T, W, H]
        path_map = torch.zeros((1, self.max_timesteps, self.grid_size[0], self.grid_size[1]), dtype=torch.float32)
        
        # Fill in the path map for each timestep
        for agent_id, agent_path in enumerate(item['paths']):
            agent_value = agent_id + 1  # 使用离散的智能体ID (1, 2, 3, ...)
            
            for t in range(self.max_timesteps):
                if t < len(agent_path):
                    r, c = agent_path[t]
                    if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                        path_map[0, t, r, c] = agent_value
        
        starts = torch.tensor(item['starts'])
        goals = torch.tensor(item['goals'])
        
        # Expand obstacle map to match path map dimensions [1, T, W, H]
        obstacle_map = obstacle_map.expand(1, self.max_timesteps, -1, -1)
        
        return obstacle_map, path_map, starts, goals


def get_dataloaders(
    data_root="data",
    grid_size=(32, 32), 
    train_samples=None, 
    val_samples=None,
    batch_size=32,
    num_workers=4,
    num_agents=16,
    use_3d=False,
    max_timesteps=32,
    train_val_split=0.8,
    **kwargs
):
    """获取训练和验证数据加载器"""
    # 计算总样本需求
    total_needed = (train_samples or 1000) + (val_samples or 200)
    
    if use_3d:
        print(f"Loading 3D dataset with {num_agents} agents...")
        full_dataset = MultiAgent3DPathPlanningDataset(
            data_root=data_root,
            grid_size=grid_size, 
            num_agents=num_agents, 
            num_samples=total_needed,
            max_timesteps=max_timesteps, 
            **kwargs
        )
    else:
        print(f"Loading multi-agent dataset with {num_agents} agents...")
        full_dataset = MultiAgentPathPlanningDataset(
            data_root=data_root,
            grid_size=grid_size, 
            num_agents=num_agents, 
            num_samples=total_needed,
            max_timesteps=max_timesteps,
            **kwargs
        )
    
    # 分割训练和验证集
    total_samples = len(full_dataset.data)
    train_size = train_samples or int(train_val_split * total_samples)
    val_size = val_samples or (total_samples - train_size)
    
    # 确保不超过可用数据
    train_size = min(train_size, total_samples)
    val_size = min(val_size, total_samples - train_size)
    
    # 分割数据
    train_data = full_dataset.data[:train_size]
    val_data = full_dataset.data[train_size:train_size+val_size]
    
    # 创建训练和验证数据集（复制类属性但不重新加载数据）
    if use_3d:
        train_dataset = MultiAgent3DPathPlanningDataset.__new__(MultiAgent3DPathPlanningDataset)
        val_dataset = MultiAgent3DPathPlanningDataset.__new__(MultiAgent3DPathPlanningDataset)
    else:
        train_dataset = MultiAgentPathPlanningDataset.__new__(MultiAgentPathPlanningDataset)
        val_dataset = MultiAgentPathPlanningDataset.__new__(MultiAgentPathPlanningDataset)
    
    # 复制原始数据集的属性
    for dataset in [train_dataset, val_dataset]:
        dataset.data_root = full_dataset.data_root
        dataset.grid_size = full_dataset.grid_size
        dataset.num_agents = full_dataset.num_agents
        dataset.num_samples = full_dataset.num_samples
        dataset.max_timesteps = full_dataset.max_timesteps
    
    # 分别设置数据
    train_dataset.data = train_data
    val_dataset.data = val_data
    
    print(f"Training samples: {len(train_dataset.data)}")
    print(f"Validation samples: {len(val_dataset.data)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def visualize_sample(path_map, obstacle_map=None, start=None, goal=None, ax=None):
    """可视化样本"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制路径
    # print(path_map.max())
    ax.imshow(path_map.squeeze() > 0.5, cmap='Greens', alpha=0.7, vmin=0, vmax=1)
    
    # 绘制障碍物
    if obstacle_map is not None:
        ax.imshow(obstacle_map.squeeze(), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    
    # 绘制起点和终点
    if start is not None:
        ax.plot(start[1], start[0], 'bo', markersize=10, label='Start')
    if goal is not None:
        ax.plot(goal[1], goal[0], 'go', markersize=10, label='Goal')
        
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    
    return ax 

def visualize_sample_process(x_seq,
                      obstacle_map=None,
                      start=None,
                      goal=None,
                      out_path='sampling.gif',
                      fps=5,
                      cmap='Greens',
                      obstacle_cmap='Reds'):
    """
    将一系列 x_t (浮点 [B,1,W,H] 或 [1,1,W,H]) 可视化，并保存为 GIF
    
    Args:
        x_seq (list of torch.Tensor or np.ndarray):
            length T list, 每个元素形状为 (B,1,W,H)
        obstacle_map (torch.Tensor or np.ndarray, optional):
            (B,1,W,H) 的同尺寸障碍物掩码
        start, goal (tuple of int):
            (row, col) 坐标
        out_path (str):
            输出 GIF 路径
        fps (int):
            帧率（帧/秒）
        cmap, obstacle_cmap (str):
            matplotlib colormap 名称
    """
    
    
    frames = []
    B = x_seq[0].shape[0]
    assert B == 1, "目前仅支持单 batch 大小 =1 的可视化"
    
    for t, x_t in enumerate(x_seq):
        
        fig, ax = plt.subplots(figsize=(6,6))
        
        # path overlay
        ax.imshow(x_t.squeeze() > 0.5, cmap=cmap, alpha=0.7, vmin=0, vmax=1)
        
        # obstacle overlay
        if obstacle_map is not None:
            ax.imshow(obstacle_map.squeeze(), cmap=obstacle_cmap, alpha=0.5, vmin=0, vmax=1)
        
        # start/goal
        if start is not None:
            ax.plot(start[1], start[0], 'bo', ms=10)
        if goal is not None:
            ax.plot(goal[1], goal[0], 'go', ms=10)
        
        ax.set_title(f"timestep {t}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # grab frame
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        h, w = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)
        plt.close(fig)
    
    # write gif
    return frames


def visualize_multi_sample_process(x_seq,
                      obstacle_map=None,
                      starts=None,
                      goals=None,
                      out_path='sampling.gif',
                      fps=5,
                      cmap='Greens',
                      obstacle_cmap='Reds'):
    """
    将一系列 x_t (浮点 [B,1,W,H] 或 [1,1,W,H]) 可视化，并保存为 GIF
    
    Args:
        x_seq (list of torch.Tensor or np.ndarray):
            length T list, 每个元素形状为 (B,1,W,H)
        obstacle_map (torch.Tensor or np.ndarray, optional):
            (B,1,W,H) 的同尺寸障碍物掩码
        start, goal (tuple of int):
            (row, col) 坐标
        out_path (str):
            输出 GIF 路径
        fps (int):
            帧率（帧/秒）
        cmap, obstacle_cmap (str):
            matplotlib colormap 名称
    """
    
    
    frames = []
    B = x_seq[0].shape[0]
    assert B == 1, "目前仅支持单 batch 大小 =1 的可视化"
    
    for t, x_t in enumerate(x_seq):
        
        fig, ax = plt.subplots(figsize=(6,6))
        
        arr = x_t.squeeze() 
        masked = np.ma.masked_where(arr <= 0, arr)
        ax.imshow(masked, cmap='viridis', alpha=0.8, vmin=0, vmax=1)
        ys, xs = np.where(arr > 0)
        vals    = arr[ys, xs]
        ax.scatter(
            xs, ys, 
            c=vals,          # color by normalized ID
            cmap='viridis', 
            s=10,            # <-- marker size (try 5, 10, 20, etc.)
            alpha=0.8,
            marker='o'
        )
            

        # 绘制障碍物
        if obstacle_map is not None:
            ax.imshow(obstacle_map.squeeze(), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
            
        # 绘制起点
        if starts is not None:
            # allow a single tuple
            if not isinstance(starts, (list, tuple)) or not hasattr(starts[0], '__iter__'):
                starts = [starts]
            for i, st in enumerate(starts):
                lbl = 'Start' if i == 0 else '_nolegend_'
                ax.plot(st[1], st[0], 's', color=plt.cm.viridis(i/len(starts)), markersize=10, label=lbl)

        # 绘制终点
        if goals is not None:
            if not isinstance(goals, (list, tuple)) or not hasattr(goals[0], '__iter__'):
                goals = [goals]
            for i, gl in enumerate(goals):
                lbl = 'Goal' if i == 0 else '_nolegend_'
                ax.plot(gl[1], gl[0], '^', color=plt.cm.viridis(i/len(goals)), markersize=10, label=lbl)


        
        ax.set_title(f"timestep {t}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # grab frame
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        h, w = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)
        plt.close(fig)
    
    # write gif
    return frames


def visualize_multi_sample(path_map,
                     obstacle_map=None,
                     starts=None,
                     goals=None,
                     ax=None):
    """可视化一个或多个路径样本"""
    import matplotlib.colors as mcolors


    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    # print(path_map)
    arr = path_map.squeeze() 
    masked = np.ma.masked_where(arr <= 0, arr)
    ax.imshow(masked, cmap='viridis', alpha=0.8, vmin=0, vmax=1)
    ys, xs = np.where(arr > 0)
    vals    = arr[ys, xs]
    ax.scatter(
        xs, ys, 
        c=vals,          # color by normalized ID
        cmap='viridis', 
        s=10,            # <-- marker size (try 5, 10, 20, etc.)
        alpha=0.8,
        marker='o'
    )
        

    # 绘制障碍物
    if obstacle_map is not None:
        ax.imshow(obstacle_map.squeeze(), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        
    # 绘制起点
    if starts is not None:
        # allow a single tuple
        if not isinstance(starts, (list, tuple)) or not hasattr(starts[0], '__iter__'):
            starts = [starts]
        for i, st in enumerate(starts):
            lbl = 'Start' if i == 0 else '_nolegend_'
            ax.plot(st[1], st[0], 's', color=plt.cm.viridis(i/len(starts)), markersize=10, label=lbl)

    # 绘制终点
    if goals is not None:
        if not isinstance(goals, (list, tuple)) or not hasattr(goals[0], '__iter__'):
            goals = [goals]
        for i, gl in enumerate(goals):
            lbl = 'Goal' if i == 0 else '_nolegend_'
            ax.plot(gl[1], gl[0], '^', color=plt.cm.viridis(i/len(goals)), markersize=10, label=lbl)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    return ax

def visualize_3d_sample(path_map,
                     obstacle_map=None,
                     starts=None,
                     goals=None,
                     ax=None,
                     path_cmap='viridis',
                     obstacle_color='red',
                     obstacle_alpha=0.2):
    """可视化样本"""
    
    # 绘制路径
    # print(path_map.max())
    arr = path_map.squeeze()  # [T, W, H]
    if obstacle_map is not None:
        obs_np = obstacle_map.squeeze()  # [T, W, H]
    T, W, H = arr.shape
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
    
    # draw obstacles on floor
    from matplotlib import colors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.pyplot as plt
    rgba = colors.to_rgba(obstacle_color, obstacle_alpha) 
    cmap = plt.colormaps.get_cmap(path_cmap)
    # draw obstacles on floor
    if obstacle_map is not None:
        ys_o, xs_o = np.where(obs_np[0] > 0)
        for x_o, y_o in zip(xs_o, ys_o):
            ax.bar3d(x_o, y_o, 0,
                        1, 1, T,
                        color=rgba,
                        alpha=obstacle_alpha,
                        shade=True)

    # draw path voxels at height = time
    coords = np.argwhere(arr > 0)
    if coords.size:
        times = coords[:,0]
        ys = coords[:,1]
        xs = coords[:,2]
        # Use agent ID values for distinct colors
        agent_values = arr[times, ys, xs]
        
        # 定义明显不同的颜色列表 - 最多支持20个智能体
        distinct_colors = [
            '#FF0000',  # 红色
            '#00FF00',  # 绿色
            '#0000FF',  # 蓝色
            '#FFFF00',  # 黄色
            '#FF00FF',  # 洋红色
            '#00FFFF',  # 青色
            '#FFA500',  # 橙色
            '#800080',  # 紫色
            '#008000',  # 深绿色
            '#000080',  # 深蓝色
            '#800000',  # 深红色
            '#808000',  # 橄榄色
            '#008080',  # 水鸭色
            '#C0C0C0',  # 银色
            '#808080',  # 灰色
            '#FFB6C1',  # 浅粉红色
            '#DDA0DD',  # 梅红色
            '#87CEEB',  # 天蓝色
            '#F0E68C',  # 卡其色
            '#FF6347'   # 番茄红
        ]
        
        for x_p, y_p, t_p, agent_val in zip(xs, ys, times, agent_values):
            # 根据智能体ID选择颜色
            agent_id = int(agent_val) - 1  # 转换为0开始的索引
            color = distinct_colors[agent_id % len(distinct_colors)]
            
            ax.bar3d(x_p, y_p, t_p,
                    1, 1, 1,
                    color=color,
                    edgecolor='k',
                    linewidth=0.2,
                    shade=True)

    # 定义相同的颜色列表用于起始点和目标点
    distinct_colors = [
        '#FF0000',  # 红色
        '#00FF00',  # 绿色
        '#0000FF',  # 蓝色
        '#FFFF00',  # 黄色
        '#FF00FF',  # 洋红色
        '#00FFFF',  # 青色
        '#FFA500',  # 橙色
        '#800080',  # 紫色
        '#008000',  # 深绿色
        '#000080',  # 深蓝色
        '#800000',  # 深红色
        '#808000',  # 橄榄色
        '#008080',  # 水鸭色
        '#C0C0C0',  # 银色
        '#808080',  # 灰色
        '#FFB6C1',  # 浅粉红色
        '#DDA0DD',  # 梅红色
        '#87CEEB',  # 天蓝色
        '#F0E68C',  # 卡其色
        '#FF6347'   # 番茄红
    ]
    
    # start & goal with consistent colors
    if starts is not None:
        pts = starts if hasattr(starts[0], '__iter__') else [starts]
        for i, (r, c) in enumerate(pts):
            color = distinct_colors[i % len(distinct_colors)]
            ax.bar3d(c, r, 0,
                    1, 1, 1,
                    color=color,
                    alpha=1.0,
                    edgecolor='k',
                    linewidth=2,
                    label='Start' if i==0 else None)
    if goals is not None:
        pts = goals if hasattr(goals[0], '__iter__') else [goals]
        for i, (r, c) in enumerate(pts):
            color = distinct_colors[i % len(distinct_colors)]
            ax.bar3d(c, r, T-1,  # 稍微降低一点，避免与最后一个时间步重叠
                    1, 1, 1,
                    color=color,
                    alpha=1.0,
                    edgecolor='white',  # 用白色边框区分目标点
                    linewidth=2,
                    label='Goal' if i==0 else None)

        ax.set_title(f"3d ground truth", fontsize=12)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Time')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(0, H-1); ax.set_ylim(0, W-1); ax.set_zlim(0, T-1)

        return ax

def visualize_3d_sample_process(x_seq,
                                obstacle_map=None,
                                starts=None,
                                goals=None,
                                path_cmap='viridis',
                                obstacle_color='red',
                                obstacle_alpha=0.2,
                                frame_interval=4):  # Add frame interval parameter
    """
    Visualize a 3D diffusion / path sampling process as a sequence of 3D scatter frames.

    Args:
        x_seq (list of torch.Tensor or np.ndarray):
            length S list, each element shape (B,1,T,W,H) of floats in [0,1].
        obstacle_map (torch.Tensor or np.ndarray, optional):
            shape (B,1,T,W,H), same indexing, 1 = obstacle.
        starts, goals (tuple or list of tuples, optional):
            (row, col) coordinates.
        path_cmap (str): matplotlib colormap for paths.
        obstacle_color (str): color for obstacle markers.
        obstacle_alpha (float): alpha for obstacle markers.
        frame_interval (int): interval between frames to visualize.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.cm import get_cmap

    # only support batch-size 1
    B = x_seq[0].shape[0]
    assert B == 1, "batch size must be 1"
    if obstacle_map is not None:
        obs_np = obstacle_map.squeeze()  # [T,W,H]
    cmap = get_cmap(path_cmap)

    frames = []
    for frame_idx, x_t in enumerate(x_seq):
        # Only generate frames at specified intervals
        if frame_idx % frame_interval != 0 and frame_idx != len(x_seq) - 1:
            continue
            
        # x_t: (1,T,W,H) or numpy
        arr = x_t.squeeze()   # [T,W,H]
        T, W, H = arr.shape

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        # draw obstacles on floor
        from matplotlib import colors
        rgba = colors.to_rgba(obstacle_color, obstacle_alpha) 
        if obstacle_map is not None:
            ys_o, xs_o = np.where(obs_np[0] > 0)
            for x_o, y_o in zip(xs_o, ys_o):
                ax.bar3d(x_o, y_o, 0,
                            1, 1, T,
                            color=rgba,
                            alpha=obstacle_alpha,
                            shade=True)

        # draw path voxels at height = time
        coords = np.argwhere(arr > 0)
        if coords.size:
            times = coords[:,0]
            ys = coords[:,1]
            xs = coords[:,2]
            values = arr[times, ys, xs]
            colors = cmap(values)
            for x_p, y_p, t_p, c_p in zip(xs, ys, times, colors):
                ax.bar3d(x_p, y_p, t_p,
                         1, 1, 1,
                         color=c_p,
                         edgecolor='k',
                         linewidth=0.2,
                         shade=True)

        # start & goal
        if starts is not None:
            pts = starts if hasattr(starts[0], '__iter__') else [starts]
            for i, (r, c) in enumerate(pts):
                ax.bar3d(c, r, 0,
                         1, 1, 1,
                         color=plt.cm.viridis((i+1)/len(starts)),
                         alpha=1.0,
                         edgecolor='k',
                         linewidth=1,
                         label='Start' if i==0 else None)
        if goals is not None:
            pts = goals if hasattr(goals[0], '__iter__') else [goals]
            for i, (r, c) in enumerate(pts):
                ax.bar3d(c, r, T,
                         1, 1, 1,
                         color=plt.cm.viridis((i+1)/len(goals)),
                         alpha=1.0,
                         edgecolor='k',
                         linewidth=1,
                         label='Goal' if i==0 else None)

        ax.set_title(f"Step {frame_idx}", fontsize=12)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Time')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(0, H-1); ax.set_ylim(0, W-1); ax.set_zlim(0, T-1)

        # grab the frame as an RGB array
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        h, w = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)
        plt.close(fig)

    return frames



def visualize_3d_sample_in_2d(path_map,
                     obstacle_map=None,
                     starts=None,
                     goals=None,
                     ax=None,
                     return_frames=False):
    """可视化2D轨迹样本"""
    
    # Get data for all timesteps
    arr = path_map.squeeze()  # [T, W, H]
    if obstacle_map is not None:
        obs = obstacle_map.squeeze()  # [T, W, H]
    T, W, H = arr.shape
  
    


    frames = []

    for t in range(T):
        # 每帧都新建 Figure/ax，确保干净
        fig, ax = plt.subplots(figsize=(6,6))

        # 绘制路径
        path_t = arr[t]
        masked = np.ma.masked_where(path_t <= 0, path_t)
        ax.imshow(masked, cmap='viridis', alpha=0.8, vmin=0, vmax=1)
        ys, xs = np.where(path_t > 0)
        vals    = path_t[ys, xs]
        ax.scatter(
            xs, ys, 
            c=vals,          # color by normalized ID
            cmap='viridis', 
            s=10,            # <-- marker size (try 5, 10, 20, etc.)
            alpha=0.8,
            marker='o'
        )
            

        # 绘制障碍物
        obs_t = obs[t]
        if obs_t is not None:
            ax.imshow(obs_t, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
            
        # 绘制起点
        if starts is not None:
            # allow a single tuple
            if not isinstance(starts, (list, tuple)) or not hasattr(starts[0], '__iter__'):
                starts = [starts]   
            for i, st in enumerate(starts):
                lbl = 'Start' if i == 0 else '_nolegend_'
                ax.plot(st[1], st[0], 's', color=plt.cm.viridis((i+1)/len(starts)), markersize=10, label=lbl)

        # 绘制终点
        if goals is not None:
            if not isinstance(goals, (list, tuple)) or not hasattr(goals[0], '__iter__'):
                goals = [goals]
            for i, gl in enumerate(goals):
                lbl = 'Goal' if i == 0 else '_nolegend_'
                ax.plot(gl[1], gl[0], '^', color=plt.cm.viridis((i+1)/len(goals)), markersize=10, label=lbl)

        ax.set_aspect('equal')
        
        ax.set_xticks([])
        ax.set_yticks([])
        # if (starts is not None) or (goals is not None):
        #     ax.legend(loc='upper right')
        ax.set_title(f'Timestep {t}')

        # 抓取图像数据
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        h, w = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)
        plt.close(fig)

    return frames


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='数据根目录')
    parser.add_argument('--use_3d', action='store_true', help='是否使用3D多智能体数据集')
    parser.add_argument('--num_agents', type=int, default=16, help='智能体数量')
    parser.add_argument('--train_samples', type=int, default=None, help='训练样本数量限制')
    parser.add_argument('--val_samples', type=int, default=None, help='验证样本数量限制')
    parser.add_argument('--grid_width', type=int, default=32, help='网格宽度')
    parser.add_argument('--grid_height', type=int, default=32, help='网格高度')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--max_timesteps', type=int, default=48, help='最大时间步数')
    args = parser.parse_args()

    grid_size = (args.grid_width, args.grid_height)
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        grid_size=grid_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        num_agents=args.num_agents,
        use_3d=args.use_3d,
        max_timesteps=args.max_timesteps
    )

    print('可视化部分样本并保存到图片...')
    
    # 设置matplotlib后端为Agg，不显示图形界面
    import matplotlib
    matplotlib.use('Agg')
    
    # 创建输出目录
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, batch in enumerate(val_loader):
        obstacle_map, path_maps, starts, goals = batch
        
        if args.use_3d:
            # 3D: obs [B,1,T,H,W], paths [B,1,T,H,W]
            B, _, T, H, W = path_maps.shape
            
            for j in range(min(B, 2)):  # 只看前两个样本
                starts_list = [tuple(starts[j, k].numpy()) for k in range(starts.shape[1])]
                goals_list  = [tuple(goals[j, k].numpy())  for k in range(goals.shape[1])]

                # 保存带障碍物的3D可视化图（更透明）
                fig1 = plt.figure(figsize=(10, 8))
                ax1 = fig1.add_subplot(111, projection='3d')
                
                visualize_3d_sample(
                    path_map=path_maps[j].numpy(),  # [1,T,H,W]
                    obstacle_map=obstacle_map[j].numpy(),  # [1,T,H,W]
                    starts=starts_list,
                    goals=goals_list,
                    ax=ax1,
                    obstacle_alpha=0.1  # 更透明的障碍物
                )
                
                # 设置45度视角
                ax1.view_init(elev=30, azim=45)
                
                # 保存带障碍物的图片
                output_path_with_obs = os.path.join(output_dir, f"3d_sample_{j}_with_obstacles.png")
                plt.savefig(output_path_with_obs, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved 3D visualization with obstacles to: {output_path_with_obs}")
                
                # 保存不带障碍物的3D可视化图
                fig2 = plt.figure(figsize=(10, 8))
                ax2 = fig2.add_subplot(111, projection='3d')
                
                visualize_3d_sample(
                    path_map=path_maps[j].numpy(),  # [1,T,H,W]
                    obstacle_map=None,  # 不显示障碍物
                    starts=starts_list,
                    goals=goals_list,
                    ax=ax2
                )
                
                # 设置45度视角
                ax2.view_init(elev=30, azim=45)
                
                # 保存不带障碍物的图片
                output_path_no_obs = os.path.join(output_dir, f"3d_sample_{j}_no_obstacles.png")
                plt.savefig(output_path_no_obs, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved 3D visualization without obstacles to: {output_path_no_obs}")
        else:
            # 2D multi-agent: obs [B,1,H,W], paths [B,1,H,W]
            B, _, H, W = path_maps.shape

            for j in range(min(B, 2)):  # 只看前两个样本
                starts_list = [tuple(starts[j, k].numpy()) for k in range(starts.shape[1])]
                goals_list  = [tuple(goals[j, k].numpy())  for k in range(goals.shape[1])]

                fig, ax = plt.subplots(figsize=(8, 8))
                visualize_multi_sample(
                    path_map=path_maps[j, 0].numpy(),  # [H,W]
                    obstacle_map=obstacle_map[j,0].numpy(),  # [H,W]
                    starts=starts_list,
                    goals=goals_list,
                    ax=ax,
                )
                
                # 保存图片
                output_path = os.path.join(output_dir, f"2d_sample_{j}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved 2D visualization to: {output_path}")

        break
