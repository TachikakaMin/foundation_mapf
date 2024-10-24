import torch
import cv2
import os
import numpy as np
from args import get_args
from data_preprocess.datasetMAPF import MAPFDataset
from path_visualization import sample_agent_action_update, calculate_current_goal_distance

def read_test_sample(sample_path, agent_idx_dim):
    data = MAPFDataset(sample_path, agent_idx_dim)
    start_location = data[0]
    
    feature = start_location["feature"]  # shape:[channel_len, n, m]
    mask = start_location["mask"]  # shape:[n, m]
    agent_num = torch.sum(mask == 1)
    
    _map = feature[0, :, :]  # shape:[n, m]
    channel_len = feature.shape[0]
    agent_idx_len = (channel_len - 1) // 2  
    current_loc = feature[-agent_idx_len:, :, :] # shape:[agent_idx_len, n, m]
    goal_loc = feature[1:agent_idx_len+1, :, :] # shape:[agent_idx_len, n, m]
    
    current_loc_tuple = torch.nonzero(mask, as_tuple=False) # shape:[agent_num, 2]
    goal_mask = goal_loc.any(0) # shape:[n, m]
    goal_loc_tuple = torch.nonzero(goal_mask, as_tuple=False) # shape:[agent_num, 2]
    
    goal_loc_dict = {}
    for i in range(agent_num):
        key = goal_loc[:,goal_loc_tuple[i][0], goal_loc_tuple[i][1]]
        key = tuple(key.tolist())
        value = goal_loc_tuple[i]
        goal_loc_dict[key] = value
        
    return feature, agent_num, _map, mask, current_loc, current_loc_tuple, goal_loc, goal_loc_dict


def path_formation(model_path, sample_path, agent_idx_dim, device):
    model = torch.load(model_path)

    current_feature, agent_num, _map, \
        current_mask, current_loc, current_loc_tuple, \
        goal_loc, goal_loc_dict = read_test_sample(sample_path, agent_idx_dim)
    
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



def animate_paths(trajectories, goal_positions, _map, interval=500, save_path="test/test_path_video.mp4"):
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
        ax.text(goal[1], goal[0], f'{i}', color='black', fontsize=5, ha='center', va='center')  # Add goal index

    # Initialize agent markers for each agent at their starting positions
    agent_plots = [ax.plot([], [], 'go', markersize=10)[0] for _ in range(len(trajectories))]  # Dots for agents
    # Initialize text labels for each agent (for agent indexes on agents)
    agent_texts = [ax.text(0, 0, '', color="black", fontsize=5, ha='center', va='center') for _ in range(len(trajectories))]
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
    
    # Save the animation
    ani.save(save_path, writer=FFwriter)

    # Close the figure after saving
    plt.close(fig)
    
    



if __name__ == '__main__':
    model_path = 'path/to/model.pth'  
    sample_path = 'data/sample.yaml' 
    args = get_args()
    agent_idx_dim = int(np.ceil(np.log2(args.max_agent_num)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    current_goal_distance, _map, trajectories, goal_positions = path_formation(model_path, sample_path, agent_idx_dim, device)
    animate_paths(trajectories, goal_positions, _map, interval=500)
    