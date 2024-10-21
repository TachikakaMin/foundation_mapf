import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import itertools
from args import get_args
from models.unet import UNet
from data_preprocess.datasetMAPF import MAPFDataset

def train(args, model, train_loader, val_loader, optimizer, loss_fn, device='cuda'):
    """
    Trains the UNet model using masked loss, gradient clipping, and custom optimizer.
    Also evaluates on validation set after each epoch.

    Args:
        args: Argument object that contains training configurations like learning rate and epochs.
        model: The neural network model (UNet).
        train_loader: Dataloader for the training dataset.
        val_loader: Dataloader for the validation dataset.
        optimizer: Optimizer for training (optional).
        loss_fn: Loss function (optional, default is CrossEntropyLoss with reduction="none").
        device: Device to run the training on (default is 'cuda').
    """
    model.to(device)
    
    for epoch in range(1, args.epochs + 1):
        # Set model to training mode
        model.train()  
        for batch in tqdm(train_loader):
            # Load data onto the correct device (CPU/GPU)
            feature = batch["feature"].to(device)  # shape:[batch_size, channel_num, n, m]
            action_y = batch["action"].to(device)  # shape:[batch_size, n, m]
            mask = batch["mask"].to(device)  # shape:[batch_size, n, m]

            # Forward pass
            pred = model(feature)  # shape:[batch_size, action_dim, n, m]
            
            # Compute loss and apply mask
            loss = loss_fn(pred, action_y)  # shape:[batch_size, n, m]
            loss = loss * mask.float() # shape:[batch_size, n, m]
            averaged_loss = loss.mean(dim=0)  # shape:[n, m]
            max_loss = averaged_loss.max()  # scalar 
            
            # Backward pass and optimization
            optimizer.zero_grad()
            max_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
        # Train loss for last batch in a epoch
        train_loss = max_loss.item()
        print(f"Epoch {epoch}/{args.epochs}, Training Loss: {train_loss}")

        # Evaluate on validation set
        val_loss = evaluate_valid_loss(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs}, Validation Loss: {val_loss}")
        
        # Evaluate path finding
        sample_agent_path_animation(model, val_loader, a=0, b=0, device=device, steps=100)
        

def evaluate_valid_loss(model, val_loader, loss_fn, device='cuda'):
    """
    Evaluates the model on the validation set.

    Args:
        args: Argument object that contains configurations like learning rate and epochs.
        model: The neural network model (UNet).
        val_loader: Dataloader for the validation dataset.
        loss_fn: Loss function.
        device: Device to run the evaluation on (default is 'cuda').

    Returns:
        val_loss (float): The average validation loss for the entire validation set.
    """
    # Set model to evaluation mode
    model.eval()  
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for batch in val_loader:
            # Load validation data onto the correct device (CPU/GPU)
            feature = batch["feature"].to(device)
            action_y = batch["action"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            pred = model(feature)

            # Compute the loss and apply mask
            loss = loss_fn(pred, action_y)
            loss = loss * mask.float()
            averaged_loss = loss.mean(dim=0)
            max_loss = averaged_loss.max()
            val_loss += max_loss.item()

    val_loss /= len(val_loader)  
    return val_loss


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


def sample_agent_path_animation(model, val_loader, a, b, device, steps=100):
    global anim  # 声明使用全局变量
    anim = None

    global sample_feature, sample_curr_mask, sample_current_loc, current_loc_tuple
    sample_feature, sample_agent_num, sample_map, sample_curr_mask, sample_current_loc, current_loc_tuple, sample_goal_loc, goal_loc_dict = sample_agent_information(val_loader, a, b)
    
    fig, ax = plt.subplots()
    # 创建动画，调用更新函数
    anim = FuncAnimation(
        fig, 
        plot_update,  # 函数，用于在每一帧中更新图形内容
        frames=steps, # 指定动画的总帧数
        fargs=(ax, model, sample_agent_num, sample_map, sample_goal_loc, goal_loc_dict, device),  # 在调用 plot_update 时需要传递的额外参数
        interval=500, 
        repeat=False 
    )
    
    plt.show(block=False)  # 非阻塞显示动画
    plt.pause(30)  # 暂停 3 秒（根据需要调整时间）
    plt.close(fig)  # 自动关闭图形窗口


def plot_update(frame, ax, model, agent_num, map_grid, goal_loc, goal_locs_dic, device):
    global anim  # 在这里再次声明使用全局变量
    global sample_feature, sample_curr_mask, sample_current_loc, current_loc_tuple
    # 清除之前的绘图
    ax.clear()  
    # 使用颜色映射，0对应白色，1对应灰色
    cmap = ListedColormap(['white', 'gray'])
    bounds = [0, 0.5, 1]  # 数值区间 [0, 0.5) 将映射到第一个颜色（即白色），数值区间 [0.5, 1] 将映射到第二个颜色（即灰色）。
    norm = BoundaryNorm(bounds, cmap.N)  # 根据这些边界，将数值映射到颜色图中的相应颜色。
    
    # 画出地图
    ax.imshow(map_grid.numpy(), cmap=cmap, norm=norm)
    
    # 设置网格线
    ax.grid(which='both', color='black', linestyle='-', linewidth=1)
    
    # 设置刻度线在网格线的中间
    n = map_grid.shape[0] # 行数
    m = map_grid.shape[1] # 列数
    ax.set_xticks(np.arange(-.5, m, 1)) 
    ax.set_yticks(np.arange(-.5, n, 1))
    
    # 隐藏刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 给每个 agent 的起点和终点上色
    agents_colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow', 'cyan', 
        'magenta', 'brown', 'lime', 'gold', 'indigo', 'violet', 'coral', 
        'teal', 'navy', 'salmon', 'olive', 'chocolate', 'crimson', 'maroon'
    ] 
    for i, idx in enumerate(current_loc_tuple):
        cur_x, cur_y = idx 
        cur_x = cur_x.item()
        cur_y = cur_y.item()
        agent_index = sample_current_loc[:, cur_x, cur_y]
        agent_index = tuple(agent_index.tolist())
        goal_x, goal_y = goal_locs_dic[agent_index]
        goal_x = goal_x.item()
        goal_y = goal_y.item()
        color = agents_colors[i % len(agents_colors)]  # 使用不同颜色，循环使用
        # 起点（实心圆圈）
        ax.plot(cur_y, cur_x, marker='o', color=color, markersize=5)
        # 终点（空心圆圈）
        ax.plot(goal_y, goal_x, marker='o', color=color, markersize=5, markerfacecolor='none', markeredgewidth=2)
    
    
    # 计算当前目标距离
    current_goal_distance = calculate_current_goal_distance(sample_current_loc, current_loc_tuple, goal_locs_dic)
    # 如果当前目标距离为0，停止动画
    if current_goal_distance == 0:
        anim.event_source.stop()  # 停止动画
    
    
    # 更新 agent 的位置
    sample_feature, sample_curr_mask, sample_current_loc, current_loc_tuple = sample_agent_action_update(
        model, sample_feature, agent_num, map_grid, sample_curr_mask, sample_current_loc, current_loc_tuple, goal_loc,device
    )


def sample_agent_action_update(model, feature, agent_num, _map, curr_mask, current_loc, current_loc_tuple, goal_loc, device):
    model.eval()
    
    in_feature = feature.unsqueeze(0).to(device) # 增加 batch 维度; shape:[1, channel_len, n, m]
    with torch.no_grad():
        pred = model(in_feature) # shape:[1, action_dim, n, m]
    
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
    
    new_map = _map.unsqueeze(0)  # 增加 batch 维度; shape:[1, n, m]
    feature = torch.cat([new_map, goal_loc, current_loc], dim=0)
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


if __name__ == "__main__":
    # arguments
    args = get_args() 
    agent_idx_dim = int(np.ceil(np.log2(args.max_agent_num)))
    feature_channels = agent_idx_dim * 2 + 1
    
    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=feature_channels, n_classes=args.action_dim, bilinear=False)
    optimizer = torch.optim.RMSprop(net.parameters(),
                              lr=args.lr, weight_decay=1e-8, momentum=0.999, foreach=True)
    loss_fn = nn.CrossEntropyLoss(reduction="none")  

    # dataset 
    data = MAPFDataset(args.dataset_path, agent_idx_dim)  
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(data))  # 80% training, 20% validation
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])  
    
    # dataloaders
    train_loader = DataLoader(train_data, shuffle=True,  
                              batch_size=args.batch_size,  
                              num_workers=0)
    val_loader = DataLoader(val_data, shuffle=False,  
                              batch_size=args.batch_size, 
                              num_workers=0)
    
    # train
    train(args, net, train_loader, val_loader, optimizer, loss_fn, device)