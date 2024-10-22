import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
import itertools

def sample_agent_information(val_loader, a, b):
    """
    第 a 个 batch 中第 b 个样本的智能体信息。

    """
    # 得到第 a 个 batch
    val_loader_iter = iter(val_loader)
    val_batch = next(itertools.islice(val_loader_iter, a, None))
    
    # 提取 batch 中第 b 个 sample 的信息
    val_sample_feature = val_batch["feature"][b]  # shape:[channel_len, n, m]
    val_sample_curr_mask = val_batch["mask"][b]  # shape:[n, m]
    val_sample_agent_num = torch.sum(val_sample_curr_mask == 1)
    
    val_sample_map = val_sample_feature[0, :, :]  # shape:[n, m]
    channel_len = val_sample_feature.shape[0]
    agent_idx_len = (channel_len - 1) // 2  
    val_sample_current_loc = val_sample_feature[-agent_idx_len:, :, :] # shape:[agent_idx_len, n, m]
    val_sample_goal_loc = val_sample_feature[1:agent_idx_len+1, :, :] # shape:[agent_idx_len, n, m]
    
    agents_current_loc_tuple = torch.nonzero(val_sample_curr_mask, as_tuple=False) # shape:[agent_num, 2]
    val_sample_goal_mask = val_sample_goal_loc.any(0) # shape:[n, m]
    agents_goal_loc_tuple = torch.nonzero(val_sample_goal_mask, as_tuple=False) # shape:[agent_num, 2]
    
    agents_goal_loc_dict = {}
    for i in range(val_sample_agent_num):
        key = val_sample_goal_loc[:,agents_goal_loc_tuple[i][0], agents_goal_loc_tuple[i][1]]
        key = tuple(key.tolist())
        value = agents_goal_loc_tuple[i]
        agents_goal_loc_dict[key] = value
    
    return val_sample_feature, val_sample_agent_num, val_sample_map, val_sample_curr_mask, val_sample_current_loc, agents_current_loc_tuple, val_sample_goal_loc, agents_goal_loc_dict


def sample_agent_action_update(model, feature, agent_num, _map, curr_mask, current_loc, current_loc_tuple, goal_loc, device):
    model.eval()
    curr_mask = curr_mask.to(device)
    in_feature = feature.unsqueeze(0).to(device) # 增加 batch 维度; shape:[1, channel_len, n, m]
    with torch.no_grad():
        _, pred = model(in_feature) # shape:[1, action_dim, n, m]
    
    # 选择概率最高的动作
    pred = pred.squeeze(0).permute((1, 2, 0)).argmax(-1) # shape:[n, m]
    action = pred * curr_mask # shape:[n, m]
    
    # 更新智能体的tuple位置
    fix_current_loc_tuple = 1 * current_loc_tuple
    current_loc_tuple = move_agent(agent_num, current_loc_tuple, action, _map)
    
    # 更新智能体的位置（用于模型输入）
    fix_current_loc = 1 * current_loc
    current_loc = torch.zeros_like(current_loc)
    for i in range(agent_num):
        current_x = current_loc_tuple[i][0]
        current_y = current_loc_tuple[i][1]
        pre_x = fix_current_loc_tuple[i][0]
        pre_y = fix_current_loc_tuple[i][1]
        agent_index = fix_current_loc[:, pre_x, pre_y]
        current_loc[:, current_x, current_y] = agent_index
    
    map_with_batch = _map.unsqueeze(0)  # 增加 batch 维度; shape:[1, n, m]
    feature = torch.cat([map_with_batch, goal_loc, current_loc], dim=0)
    curr_mask = current_loc.any(0)

    return feature, curr_mask, current_loc, current_loc_tuple


def move_agent(agent_num, current_locs, action, _map):
    n = _map.shape[0]
    m = _map.shape[1]
    action = action.detach().cpu().numpy() # shape:[n, m]
    tmp_current_locs = 1 * current_locs
    
    # 遍历每个智能体，根据动作更新其位置
    for i in range(agent_num):
        location = tmp_current_locs[i]
        act_dir = action[location[0], location[1]]
        
        if act_dir == 0:  # left
            location[1] = max(location[1] - 1, 0)  # 向左，确保不越界
        if act_dir == 1:  # right
            location[1] = min(location[1] + 1, m - 1)  # 向右，确保不越界
        if act_dir == 2:  # up
            location[0] = max(location[0] - 1, 0)  # 向上，确保不越界
        if act_dir == 3:  # down
            location[0] = min(location[0] + 1, n - 1)  # 向下，确保不越界

        tmp_current_locs[i] = location
    
    # 处理智能体之间的碰撞
    while True:
        clash = 1
        map_mark = 1 * _map  # 创建占位地图
        for i in range(agent_num):
            location = tmp_current_locs[i]
            map_mark[location[0], location[1]] += 1  
        
        for i in range(agent_num):
            location = tmp_current_locs[i]
            if map_mark[location[0], location[1]] > 1:  # 发生碰撞
                tmp_current_locs[i] = current_locs[i]  # 回到原位置
                clash = 0  # 表示有冲突，需要继续检测
        
        if clash:  # 如果没有冲突，跳出循环
            break
    
    return tmp_current_locs





    
def calculate_current_goal_distance(current_loc, current_loc_tuple, goal_loc_dic):
    total_distance = 0
    for idx in current_loc_tuple:
        cur_x, cur_y = idx # 当前智能体的位置
        cur_x = cur_x.item()
        cur_y = cur_y.item()
        agent_index = current_loc[:, cur_x, cur_y]
        agent_index = tuple(agent_index.tolist())
        goal_x, goal_y = goal_loc_dic[agent_index]
        goal_x = goal_x.item()
        goal_y = goal_y.item()
        distance = abs(cur_x-goal_x) + abs(cur_y - goal_y)
        total_distance += distance

    return total_distance



def path_formation(model, val_loader, a, b, device):
    current_feature, agent_num, _map, \
        current_mask, current_loc, current_loc_tuple, \
        goal_loc, goal_loc_dict = sample_agent_information(val_loader, a, b)
    
    # 用于存储每个智能体在每个步骤的位置，添加初始位置
    trajectories = [ [tuple(current_loc_tuple[i].tolist())] for i in range(agent_num)]
    
    for step in range(100):
        current_feature, current_mask, current_loc, current_loc_tuple = sample_agent_action_update(
            model, current_feature, agent_num, _map, current_mask, current_loc, current_loc_tuple, goal_loc, device
        )
        # 记录当前步骤每个智能体的位置
        for i in range(agent_num):
            trajectories[i].append(tuple(current_loc_tuple[i].tolist()))

        current_goal_distance = calculate_current_goal_distance(current_loc, current_loc_tuple, goal_loc_dict)
        if current_goal_distance == 0:
            break
    
    # 记录终点位置
    goal_positions = [None] * agent_num  
    for i, idx in enumerate(current_loc_tuple):
        cur_x, cur_y = idx 
        cur_x = cur_x.item()
        cur_y = cur_y.item()
        agent_index = current_loc[:, cur_x, cur_y]
        agent_index = tuple(agent_index.tolist())
        goal_x, goal_y = goal_loc_dict[agent_index]
        goal_x = goal_x.item()
        goal_y = goal_y.item()
        goal_positions[i] = (goal_x, goal_y)
    
    return current_goal_distance, _map, trajectories, goal_positions
        

def animate_paths(trajectories, goal_positions, _map, interval=500, save_path="eval_traj/animation.gif"):
    """
    Function to create and save an animation of agents' movement towards their goal locations without trajectory lines.
    
    Args:
    - trajectories: List of lists of tuples representing agent movements over time.
    - goal_positions: List of tuples representing the final goal positions of agents.
    - _map: The grid map where the agents move.
    - interval: Time interval between frames in milliseconds.
    - save_path: Path to save the animation file (e.g., "animation.mp4" or "animation.gif").
    """
    # Map dimensions
    n, m = _map.shape
    fig, ax = plt.subplots()

    # Set up the map visualization
    cmap = ListedColormap(['white', 'grey'])  # White for free space, grey for obstacles
    norm = BoundaryNorm([0, 0.5, 1], cmap.N)
    ax.imshow(_map, cmap=cmap, norm=norm)
    
    # Add gridlines with black color
    ax.set_xticks(np.arange(0, m, 1))
    ax.set_yticks(np.arange(0, n, 1))
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    # Plot goal positions and add index labels for goals
    for i, goal in enumerate(goal_positions):
        ax.plot(goal[1], goal[0], 'bo')  # dots represent the goals
        ax.text(goal[1], goal[0], f'{i}', color='black', fontsize=5, ha='center', va='center')  # Add goal index

    # Initialize agent markers for each agent at their starting positions
    agent_plots = [ax.plot([], [], 'go')[0] for _ in range(len(trajectories))]  # dots for agents
    # Initialize text labels for each agent (for agent indexes on agents)
    agent_texts = [ax.text(0, 0, '', color="black", fontsize=5, ha='center', va='center') for _ in range(len(trajectories))]

    # Define the initialization function
    def init():
        for agent_plot, agent_text in zip(agent_plots, agent_texts):
            agent_plot.set_data([], [])  # Initialize agent positions
            agent_text.set_position((-10, -10))  # Initialize text labels offscreen
        return agent_plots + agent_texts
    
    # Define the update function for animation
    def update(frame):
        for i, trajectory in enumerate(trajectories):
            if frame < len(trajectory):
                # Update agent position
                agent_plots[i].set_data(trajectory[frame][1], trajectory[frame][0])  # Set agent position (x, y)

                # Update text label (agent index number)
                agent_texts[i].set_position((trajectory[frame][1], trajectory[frame][0]))  # Set text near agent
                agent_texts[i].set_text(f'{i}')  # Display agent index
        

        return agent_plots + agent_texts

    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Hide minor ticks (remove small stick out of the edge)
    ax.tick_params(which='both', bottom=False, left=False, right=False, top=False)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(trajectories[0]), init_func=init, blit=True, interval=interval)

    # Save the animation
    ani.save(save_path, writer='imagemagick', fps=500)  # Use 'imagemagick' for GIF or 'ffmpeg' for MP4

    # Close the figure after saving
    plt.close(fig)
