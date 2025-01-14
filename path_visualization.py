import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
import itertools
import cv2
from tqdm import tqdm

def sample_agent_information(args, val_loader, a, b):
    """
    第 a 个 batch 中第 b 个样本的智能体信息。

    """
    # # 得到第 a 个 batch
    # val_loader_iter = iter(val_loader)
    # val_batch = next(itertools.islice(val_loader_iter, a, None))
    
    batch_size = val_loader.batch_size
    start_idx = a * batch_size + b
    val_batch = val_loader.dataset[start_idx]
    map_name = val_loader.dataset.train_data_map_name[start_idx]
    dis_map = val_loader.dataset.all_distance_maps[map_name]
    # 提取 batch 中第 b 个 sample 的信息
    val_sample_feature = val_batch["feature"]  # shape:[channel_len, n, m]
    val_sample_curr_mask = val_batch["mask"]  # shape:[n, m]
    val_sample_agent_num = torch.sum(val_sample_curr_mask == 1)
    
    val_sample_map = val_sample_feature[0]  # shape:[n, m]
    val_sample_current_loc = val_sample_feature[1] # shape:[n, m]
    val_sample_goal_loc = val_sample_feature[2] # shape:[n, m]
    
    agents_current_loc_tuple = torch.nonzero(val_sample_curr_mask, as_tuple=False) # shape:[agent_num, 2]
    agents_goal_loc_tuple = torch.nonzero(val_sample_goal_loc, as_tuple=False) # shape:[agent_num, 2]
    
    agents_goal_loc_dict = {}
    for i in range(val_sample_agent_num):
        key = val_sample_goal_loc[agents_goal_loc_tuple[i][0], agents_goal_loc_tuple[i][1]]
        value = agents_goal_loc_tuple[i]
        agents_goal_loc_dict[int(key.item())] = value
    
    return val_sample_feature, val_sample_agent_num, val_sample_map, \
        val_sample_curr_mask, val_sample_current_loc, agents_current_loc_tuple, \
            val_sample_goal_loc, agents_goal_loc_dict, dis_map


def sample_agent_action_update(model, feature, agent_num, _map, \
                            curr_mask, current_loc, current_loc_tuple, \
                                goal_loc, goal_loc_dict, device, action_choice, temperature, dis_map):
    model.eval()
    m, n = curr_mask.shape
    curr_mask = curr_mask.to(device)
    in_feature = feature.unsqueeze(0).to(device) # 增加 batch 维度; shape:[1, channel_len, n, m]
    with torch.no_grad():
        logits, pred = model(in_feature) # shape:[1, action_dim, n, m]
    
    # 选择概率最高的动作
    if action_choice == "sample":
        # Get logits in the right shape
        logits_reshaped = logits[0].permute(1, 2, 0).contiguous()  # shape: [m, n, action_dim]
        
        # Apply temperature scaling only at agent positions
        for i in range(agent_num):
            current_x = current_loc_tuple[i][0]
            current_y = current_loc_tuple[i][1]
            # Scale logits for this agent's position by its temperature
            logits_reshaped[current_x, current_y] = logits_reshaped[current_x, current_y] / temperature[i]
        
        # Convert to probabilities and sample
        probs = torch.softmax(logits_reshaped.view(-1, logits.shape[1]), dim=-1)
        sampled_actions = torch.multinomial(probs, num_samples=1)
        pred = sampled_actions.view(pred.shape[2], pred.shape[3])    
    else:
        pred = pred.squeeze(0).permute((1, 2, 0)).argmax(-1) # shape:[n, m]
    action = pred * curr_mask # shape:[n, m]
    
    # 更新智能体的tuple位置
    fix_current_loc_tuple = 1 * current_loc_tuple
    current_loc_tuple, temperature = move_agent(agent_num, current_loc_tuple, action, _map, temperature)
    
    # 更新智能体的位置（用于模型输入）
    fix_current_loc = 1 * current_loc
    current_loc = torch.zeros_like(current_loc)
    for i in range(agent_num):
        current_x = current_loc_tuple[i][0]
        current_y = current_loc_tuple[i][1]
        pre_x = fix_current_loc_tuple[i][0]
        pre_y = fix_current_loc_tuple[i][1]
        agent_index = fix_current_loc[pre_x, pre_y]
        current_loc[current_x, current_y] = agent_index


    last_loc_1 = feature[1]
    feature = torch.zeros_like(feature)
    feature[0] = _map
    feature[1] = current_loc
    feature[2] = goal_loc
    feature[3] = last_loc_1

    for i in range(agent_num):
        agent_idx = current_loc[current_loc_tuple[i][0], current_loc_tuple[i][1]].item()
        agent_idx = int(agent_idx)
        agent_goal_loc = goal_loc_dict[agent_idx]

        # feature[4, current_loc_tuple[i][0], current_loc_tuple[i][1]] = agent_goal_loc[0] - current_loc_tuple[i][0]
        # feature[5, current_loc_tuple[i][0], current_loc_tuple[i][1]] = agent_goal_loc[1] - current_loc_tuple[i][1]
        distance_to_goal = dis_map[current_loc_tuple[i][0].item(), current_loc_tuple[i][1].item()][agent_goal_loc[0]][agent_goal_loc[1]]
        left_distance = dis_map[current_loc_tuple[i][0].item()-1, current_loc_tuple[i][1].item()][agent_goal_loc[0]][agent_goal_loc[1]] - distance_to_goal
        right_distance = dis_map[current_loc_tuple[i][0].item()+1, current_loc_tuple[i][1].item()][agent_goal_loc[0]][agent_goal_loc[1]] - distance_to_goal
        down_distance = dis_map[current_loc_tuple[i][0].item(), current_loc_tuple[i][1].item()-1][agent_goal_loc[0]][agent_goal_loc[1]] - distance_to_goal
        up_distance = dis_map[current_loc_tuple[i][0].item(), current_loc_tuple[i][1].item()+1][agent_goal_loc[0]][agent_goal_loc[1]] - distance_to_goal
        if left_distance > 0 and right_distance > 0:
            dx = 0
        elif left_distance > 0 and right_distance < 0:
            dx = -1
        elif left_distance < 0 and right_distance > 0:
            dx = 1
        else:
            dx = 1
        if down_distance > 0 and up_distance > 0:
            dy = 0
        elif down_distance > 0 and up_distance < 0:
            dy = -1
        elif down_distance < 0 and up_distance > 0:
            dy = 1
        else:
            dy = 1
        feature[4, current_loc_tuple[i][0], current_loc_tuple[i][1]] = dx
        feature[5, current_loc_tuple[i][0], current_loc_tuple[i][1]] = dy
        feature[6, current_loc_tuple[i][0], current_loc_tuple[i][1]] = distance_to_goal

    curr_mask = (current_loc > 0)

    return feature, curr_mask, current_loc, current_loc_tuple, temperature


def move_agent(agent_num, current_locs, action, _map, temperature):
    n = _map.shape[0]
    m = _map.shape[1]
    action = action.detach().cpu().numpy() # shape:[n, m]
    tmp_current_locs = 1 * current_locs
    # 遍历每个智能体，根据动作更新其位置
    for i in range(agent_num):
        location = tmp_current_locs[i]
        act_dir = action[location[0], location[1]]
        
        if act_dir == 2:  # left
            location[1] = max(location[1] - 1, 0)  # 向左，确保不越界
        if act_dir == 1:  # right
            location[1] = min(location[1] + 1, m - 1)  # 向右，确保不越界
        if act_dir == 3:  # up
            location[0] = max(location[0] - 1, 0)  # 向上，确保不越界
        if act_dir == 4:  # down
            location[0] = min(location[0] + 1, n - 1)  # 向下，确保不越界

        tmp_current_locs[i] = location
    
    collision_flag_per_agent = [False] * agent_num
    # 处理智能体之间的碰撞
    while True:
        collision_flag = False
        map_mark = 1 * _map  # 创建占位地图
        
        # First check normal collisions
        for i in range(agent_num):
            location = tmp_current_locs[i]
            map_mark[location[0], location[1]] += 1  
        
        for i in range(agent_num):
            location = tmp_current_locs[i]
            if map_mark[location[0], location[1]] > 1:  # 发生碰撞
                if current_locs[i][0] != location[0] or current_locs[i][1] != location[1]:
                    map_mark[location[0], location[1]] -= 1
                    tmp_current_locs[i] = current_locs[i]
                    collision_flag = True
                    collision_flag_per_agent[i] = True
                    temperature[i] += 1

        # Check position swaps - 重写交换检测逻辑
        for i in range(agent_num):
            for j in range(i + 1, agent_num):
                # 检查两个智能体是否发生位置交换
                if (tuple(tmp_current_locs[i]) == tuple(current_locs[j]) and 
                    tuple(tmp_current_locs[j]) == tuple(current_locs[i])):
                    # 发生交换，恢复原位置
                    tmp_current_locs[i] = current_locs[i]
                    tmp_current_locs[j] = current_locs[j]
                    collision_flag = True
                    collision_flag_per_agent[i] = True
                    collision_flag_per_agent[j] = True
                    temperature[i] += 1
                    temperature[j] += 1

        if not collision_flag:
            break
    for i in range(agent_num):
        if collision_flag_per_agent[i] is False:
            temperature[i] -= 2
            temperature[i] = max(temperature[i], 1)
    return tmp_current_locs, temperature


def calculate_current_goal_distance(current_loc, current_loc_tuple, goal_loc_dic):
    total_distance = 0
    for idx in current_loc_tuple:
        cur_x, cur_y = idx # 当前智能体的位置
        cur_x = cur_x.item()
        cur_y = cur_y.item()
        agent_index = current_loc[cur_x, cur_y]
        goal_x, goal_y = goal_loc_dic[int(agent_index.item())]
        goal_x = goal_x.item()
        goal_y = goal_y.item()
        distance = abs(cur_x-goal_x) + abs(cur_y - goal_y)
        total_distance += distance

    return total_distance


def path_formation(args, model, val_loader, a, b, device, action_choice="max"):
    current_feature, agent_num, _map, \
        current_mask, current_loc, current_loc_tuple, \
        goal_loc, goal_loc_dict, dis_map = sample_agent_information(args, val_loader, a, b)
    
    # 用于存储每个智能体在每个步骤的位置，添加初始位置
    trajectories = [ [tuple(current_loc_tuple[i].tolist())] for i in range(agent_num)]
    temperature = [1.0] * agent_num
    for step in tqdm(range(150)):
        current_feature, current_mask, current_loc, current_loc_tuple, temperature = sample_agent_action_update(
            model, current_feature, agent_num, _map, \
                current_mask, current_loc, current_loc_tuple, \
                    goal_loc, goal_loc_dict, device, action_choice, temperature, dis_map
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
        agent_index = current_loc[cur_x, cur_y]
        goal_x, goal_y = goal_loc_dict[int(agent_index.item())]
        goal_x = goal_x.item()
        goal_y = goal_y.item()
        goal_positions[i] = (goal_x, goal_y)
    
    return current_goal_distance, _map, trajectories, goal_positions
        

def animate_paths(args, name, epoch, trajectories, goal_positions, _map, interval=500, save_path="eval_traj"):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as animation
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Map dimensions
    n, m = _map.shape
    
    # Set up the figure with larger size (e.g., 10x10 inches)
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figsize to control image size

    # Set up the map visualization
    cmap = ListedColormap(['white', 'grey'])  # White for free space, grey for obstacles
    norm = BoundaryNorm([0, 0.5, 1], cmap.N)
    ax.imshow(_map, cmap=cmap, norm=norm)

    # Add gridlines with black color
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # Plot goal positions and add index labels for goals
    for i, goal in enumerate(goal_positions):
        ax.plot(goal[1], goal[0], 'bo', markersize=10)  # Dots represent the goals, size is adjusted to be large
        ax.text(goal[1], goal[0], f'{i}', color='black', fontsize=10, ha='center', va='center')  # Add goal index

    # Initialize agent markers for each agent at their starting positions
    agent_plots = [ax.plot([], [], 'go', markersize=10)[0] for _ in range(len(trajectories))]  # Dots for agents
    # Initialize text labels for each agent (for agent indexes on agents)
    agent_texts = [ax.text(0, 0, '', color="black", fontsize=10, ha='center', va='center') for _ in range(len(trajectories))]
    # Initialize arrows from each agent to its goal
    agent_arrows = [ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='red')) for _ in range(len(trajectories))]

    # Define the initialization function
    def init():
        for agent_plot, agent_text, agent_arrow in zip(agent_plots, agent_texts, agent_arrows):
            agent_plot.set_data([], [])  # Initialize agent positions
            agent_text.set_position((-10, -10))  # Initialize text labels offscreen
            agent_arrow.set_position((-10, -10))  # Initialize arrows offscreen
        return agent_plots + agent_texts + agent_arrows

    # Define the update function for animation
    def update(frame):
        for i, trajectory in enumerate(trajectories):
            if frame < len(trajectory):
                position = trajectory[frame]
            else:
                position = trajectory[-1]  # Stay at the last position

            # Set agent position (x, y)
            agent_plots[i].set_data([position[1]], [position[0]])  
            agent_texts[i].set_position((position[1], position[0]))  # Set text near agent
            agent_texts[i].set_text(f'{i}')  # Display agent index

            # Update the arrow from the agent's current position to the goal position
            goal = goal_positions[i]
            agent_arrows[i].set_position((position[1], position[0]))  # Update arrow's start point
            agent_arrows[i].xy = (goal[1], goal[0])  # Set arrow's end point

        return agent_plots + agent_texts + agent_arrows

    # Hide axis tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Adjust axis limits if necessary
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(-0.5, n - 0.5)

    # Invert Y-axis if needed
    ax.invert_yaxis()

    # Set the number of frames to the maximum trajectory length
    max_frames = max(len(traj) for traj in trajectories)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, interval=interval)
    
    fps = 5
    FFwriter = animation.FFMpegWriter(fps=fps)
    
    # 将生成的动画保存为 MP4 文件。
    save_path = os.path.join(save_path, f"{args.current_time}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f"video_{epoch}.mp4")
    # Save the animation
    ani.save(save_path, writer=FFwriter)

    # Close the figure after saving
    plt.close(fig)
    
    # Read the saved MP4 file and convert to frames for TensorBoard
    # 使用 OpenCV 读取保存的 MP4 文件，逐帧读取并转换为 RGB 格式。
    # 将读取的帧转换为 TensorBoard 兼容的格式 (batch, time, channels, height, width)。
    # 将视频写入 TensorBoard，用于训练的可视化。
    video = cv2.VideoCapture(save_path)
    frames = []
    success, frame = video.read()
    
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)
        success, frame = video.read()

    frames = np.array(frames)
    frames = np.transpose(frames, (0, 3, 1, 2))  # Convert to (1, N, C, H, W) for TensorBoard
    frames = np.expand_dims(frames, 0)
    # Log the video to TensorBoard
    args.writer.add_video(f'animation_{name}', frames,global_step=epoch, fps=fps)
