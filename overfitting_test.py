import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from args import get_args
from models.unet import UNet
from data_preprocess.datasetMAPF import MAPFDataset
from evaluation import evaluate_valid_loss
from path_visualization import path_formation, animate_paths
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')

def train(args, model, train_loader, val_loader, optimizer, loss_fn, device):
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
        train_loss = 0
        total_agent = 0
        for batch in tqdm(train_loader):
            # Load data onto the correct device (CPU/GPU)
            feature = batch["feature"].to(device)  # shape:[batch_size, channel_num, n, m]
            action_y = batch["action"].to(device)  # shape:[batch_size, n, m]
            mask = batch["mask"].to(device)  # shape:[batch_size, n, m]

            # Forward pass
            logit, _ = model(feature)  # shape:[batch_size, action_dim, n, m]
            
            # Compute loss and apply mask
            loss = loss_fn(logit, action_y)  # shape:[batch_size, n, m]
            loss = loss * mask.float() # shape:[batch_size, n, m]
            averaged_loss = loss.sum() / mask.sum() # scalar 
            
            
            # Backward pass and optimization
            optimizer.zero_grad()
            averaged_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            # Train loss for epoch
            train_loss += loss.sum().item()
            total_agent += mask.sum().item()
            
        args.writer.add_scalar('Loss/Train', train_loss/total_agent, epoch)
        print(f"Epoch {epoch}/{args.epochs}, Training mean Loss: {train_loss/total_agent}")
        
        if epoch % args.plot_interval == 0:
            # Evaluate on validation set
            val_loss = evaluate_valid_loss(model, val_loader, loss_fn, device)
            args.writer.add_scalar('Loss/Val', val_loss, epoch)
            print(f"Epoch {epoch}/{args.epochs}, Validation mean Loss: {val_loss}")
            
            # sample path visualization
            current_goal_distance, _map, trajectories, goal_positions = path_formation(model, val_loader, 0, 0, device)
            animate_paths(args, epoch, trajectories, goal_positions, _map, interval=500)
            args.writer.add_scalar('Loss/video_goal_dis', current_goal_distance, epoch)
            print(current_goal_distance)

        if epoch % args.save_interval == 0:
            file_path = os.path.join(args.real_log_dir, f"model_checkpoint_epoch_{epoch}.pth")
            model.save_model(file_path)
        
    return current_goal_distance


if __name__ == "__main__":
    
    # arguments
    args = get_args() 
    args.current_time = datetime.now().strftime("%Y%m%d-%H%M%S") # get current date and time
    args.real_log_dir = os.path.join(args.log_dir, f"{args.current_time}")
    args.writer = SummaryWriter(log_dir = args.real_log_dir) # 创建了一个 SummaryWriter 对象，并指定将日志写入到前面定义的 real_log_dir 目录中。
    args_dict = vars(args)  # 将 args 转换为字典
    args_str = '\n'.join([f'{key}: {value}' for key, value in args_dict.items()])  # 转换为字符串
    args.writer.add_text('Args', args_str, 0)  # 这一行代码将前面生成的参数字符串 args_str 记录到日志文件中。add_text() 方法用于在日志文件中添加文本信息，这里使用 Args 作为标题，并将 args_str 的内容记录在 step=0 的位置。
    
    
    agent_idx_dim = int(np.ceil(np.log2(args.max_agent_num)))
    feature_channels = agent_idx_dim * 2 + 1
    
    torch.manual_seed(args.seed)  # Set seed for torch
    np.random.seed(args.seed)     # Set seed for numpy
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)  # If using GPU, set the seed for all devices

    
    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=feature_channels, n_classes=args.action_dim, bilinear=False)
    optimizer = torch.optim.RMSprop(net.parameters(),
                              lr=args.lr, weight_decay=1e-8, momentum=0.999, foreach=True)
    loss_fn = nn.CrossEntropyLoss(reduction="none")  

    # dataset 
    data = MAPFDataset(args.dataset_path, agent_idx_dim)  
    # Split dataset into train and validation sets
    # train_size = int(0.8 * len(data))  # 80% training, 20% validation
    # val_size = len(data) - train_size
    # train_data, val_data = random_split(data, [train_size, val_size])  
    # train_data = val_data = data
    # dataloaders
    train_loader = DataLoader(data, shuffle=True,  
                              batch_size=args.batch_size,  
                              num_workers=0)
    val_loader = DataLoader(data, shuffle=False,  
                              batch_size=args.batch_size, 
                              num_workers=0)
    
    # train
    train(args, net, train_loader, val_loader, optimizer, loss_fn, device)