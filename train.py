import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import itertools
from args import get_args
from models.unet import UNet
from dataset.datasetMAPF import MAPFDataset
from evaluation import evaluate

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
        model.train()  # Set model to training mode
        #epoch_loss = 0
        for batch in tqdm(train_loader):
            # Load data onto the correct device (CPU/GPU)
            feature = batch["feature"].to(device)
            action_y = batch["action"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            pred = model(feature)
            
            # Compute loss with reduction="none" and apply mask
            loss = loss_fn(pred, action_y)  # Loss for each element
            loss = (loss * mask.float()).max()
            averaged_loss = loss.mean(dim=0)  # Averaging across the batch dimension
            max_loss = averaged_loss.max()  # Select the maximum value from the [32, 32] averaged loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            max_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            #epoch_loss += loss.item()
        # Train loss for last batch in a epoch
        train_loss = max_loss.item()
        
        print(f"Epoch {epoch}/{args.epochs}, Training Loss: {train_loss}")

        # Evaluate on validation set
        val_loss = evaluate_valid_loss(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs}, Validation Loss: {val_loss}")
        


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
    model.eval()  # Set model to evaluation mode
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
            loss = (loss * mask.float()).max()
            averaged_loss = loss.mean(dim=0)
            max_loss = averaged_loss.max()

            val_loss += max_loss.item()

    val_loss /= len(val_loader)  # Average the validation loss # len(val_loader)批次的总数
    return val_loss


def sample_agent_action_update(model, val_loader, n, m, device='cuda'):
    """
    对第 n 个 batch 中第 m 个样本的智能体位置进行更新。

    Args:
    - model: 训练好的模型。
    - val_loader: 验证集数据加载器。
    - n: 批次的索引（第 n 个 batch）。
    - m: 批次中样本的索引（第 m 个样本）。
    - device: 运行设备（默认 'cuda'）。
    """
    model.eval()
    val_loader_iter = iter(val_loader)
    # 使用 islice 跳到第 n 个 batch
    val_batch = next(itertools.islice(val_loader_iter, n - 1, None))
    
    # 提取 batch 中第 m 个 sample 的特征、掩码等信息
    val_sample_feature = val_batch["feature"][m].to(device) # (channels, h, w)
    val_sample_mask = val_batch["mask"][m].to(device) # (h, w)
    
    # 获取通道信息
    channel_len = val_sample_feature.shape[0]
    agent_len = (channel_len)-1 // 2  # 去掉障碍物通道，计算智能体的通道数量
    
    # 提取智能体当前位置和地图信息
    val_sample_current_loc = val_sample_feature[-agent_len:, :, :] # (agent_len, h, w)
    val_sample_map = val_sample_feature[0, :, :] # (h, w)
    val_sample_goal_loc = val_sample_feature[1:agent_len+1, :, :] # (agent_len, h, w)
    goal_mask = val_sample_goal_loc.any(0)
    # 获取所有非零位置（即有智能体的位置）
    goal_positions = torch.nonzero(goal_mask, as_tuple=False)  # 形状为 (n, 2) 的张量
    goal_dict = {}
    for i in range(agent_len):
        key = val_sample_goal_loc[:,goal_positions[i][0], goal_positions[i][1]]
        value = goal_positions[i]
        goal_dict[key] = value
        
    
    # 运行多次循环模拟智能体行动更新
    for i in range(100):
        in_feature = val_sample_feature.unsqueeze(0).to(device) # 增加 batch 维度，变成 (1, channels, h, w)
        with torch.no_grad():
            pred = model(in_feature)
        # 选择概率最高的动作
        pred = pred.squeeze(0).permute((1, 2, 0)).argmax(-1) # （h, w）
        action = pred * val_sample_mask # （h, w）
        val_sample_current_loc = move_agent(val_sample_current_loc, val_sample_mask, action, val_sample_map)
        val_sample_feature = torch.cat([val_sample_map, val_sample_goal_loc, val_sample_current_loc], dim=0)
        val_sample_mask = val_sample_current_loc.any(0)
        current_goal_distance = calculate_current_goal_distance(val_sample_goal_loc, goal_dict, val_sample_mask)
        if not current_goal_distance:  # 不用返回所有时间步吗？
            return 0
            #return i+1, 0  # 只需要最后能到终点就好，不需要最优时间步数，不需要和test的路径做对比，所以不用额外记录test的完整路径
    return current_goal_distance
        
        
        
        
            
                       
def move_agent(current_loc, mask, act, map_feature):
    """
    移动智能体，根据给定的动作更新智能体的位置，同时避免智能体之间的碰撞。

    """
    agent_current_loc = torch.nonzero(mask, as_tuple=False).cpu().numpy()
    fix_current_loc = 1 * agent_current_loc
    act = act.detach().cpu().numpy() # (h, w)
    agent_num = torch.sum(mask == 1)
    h = map_feature.shape[0]
    w = map_feature.shape[1]
    
    # 遍历每个智能体，根据动作更新其位置
    for i in range(agent_num):
        location = agent_current_loc[i]
        act_dir = act[location[0], location[1]]
        
        if act_dir == 0:  # left
            location[1] = max(location[1] - 1, 0)  # 向左，确保不越界
        if act_dir == 1:  # right
            location[1] = min(location[1] + 1, w - 1)  # 向右，确保不越界
        if act_dir == 2:  # up
            location[0] = max(location[0] - 1, 0)  # 向上，确保不越界
        if act_dir == 3:  # down
            location[0] = min(location[0] + 1, h - 1)  # 向下，确保不越界

        agent_current_loc[i] = location

    # 处理智能体之间的碰撞
    while True:
        clash = 1
        map_mark = 1 * map_feature  # 创建占位地图
        for i in range(agent_num):
            location = agent_current_loc[i]
            map_mark[location[0], location[1]] += 1  # 标记智能体位置
        
        for i in range(agent_num):
            location = agent_current_loc[i]
            if map_mark[location[0], location[1]] > 1:  # 发生碰撞
                agent_current_loc[i] = fix_current_loc[i]  # 回到原位置
                clash = 0  # 表示有冲突，需要继续检测
        
        if clash:  # 如果没有冲突，跳出循环
            break
    updated_current_loc = torch.zeros_like(current_loc)
    for i in range(agent_num):
        current_x = agent_current_loc[i][0]
        current_y = agent_current_loc[i][1]
        pre_x = fix_current_loc[i][0]
        pre_y = fix_current_loc[i][1]
        agent_index = current_loc[:, pre_x, pre_y]
        updated_current_loc[:, current_x, current_y] = agent_index
        
    return updated_current_loc


def calculate_current_goal_distance(current_loc, goal_dict, mask):
    """
    计算所有智能体当前位置到目标位置的曼哈顿距离总和，利用 mask 提取当前位置和目标位置。

    Args:
    - current_loc (Tensor): 智能体的当前位置，形状为 (agent_len, h, w)，one-hot 编码。
    - goal_loc (Tensor): 智能体的目标位置，形状为 (agent_len, h, w)，one-hot 编码。
    - mask (Tensor): 表示智能体位置的掩码，形状为 (h, w)，布尔值张量，True 表示有智能体。

    Returns:
    - total_distance (float): 曼哈顿距离总和。
    """
    total_distance = 0  # 初始化曼哈顿距离总和

    # 获取所有有智能体的位置
    agent_positions = torch.nonzero(mask, as_tuple=False)  # (num_agents, 2)

    # 遍历每个智能体的位置
    total_distance = 0
    for i in agent_positions:
        cur_x, cur_y = i # 当前智能体的位置
        agent_index = current_loc[:, cur_x, cur_y]
        goal_x, goal_y = goal_dict[agent_index]
        distance = torch.abs(cur_x-goal_x) + torch.abs(cur_y - goal_y)
        total_distance += distance

    return total_distance


if __name__ == "__main__":
    args = get_args() 
    # the number of binary bits can be used to represent the number of agents
    args.agent_dim = int(np.ceil(np.log2(args.max_agent_num)))
    # every grid of a input map is represented by a feature vector with feature_dim*2+1 features
    # the first feature represents the existence of obstacle
    # the next feature_dim features represents the goal position of a specific agent
    # the next feature_dim features represents the start position of a specific agent
    feature_channels = args.agent_dim * 2 + 1
    
    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=feature_channels, n_classes=args.action_dim, bilinear=False)
    optimizer = torch.optim.RMSprop(net.parameters(),
                              lr=args.lr, weight_decay=1e-8, momentum=0.999, foreach=True)
    loss_fn = nn.CrossEntropyLoss(reduction="none")  # Cross entropy loss with no reduction

    # dataset 
    data = MAPFDataset(args.dataset_path, args.agent_dim)  # Load entire dataset
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(data))  # 80% training, 20% validation
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])  # Randomly split the dataset
    
    # dataloaders
    train_loader = DataLoader(train_data, shuffle=True,  
                              batch_size=args.batch_size, 
                              num_workers=1)
    
    val_loader = DataLoader(val_data, shuffle=False,  
                              batch_size=args.batch_size, 
                              num_workers=1)

    # train
    train(args, net, train_loader, val_loader, optimizer, loss_fn, device)