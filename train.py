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
import random
matplotlib.use('Agg')

def train(args, model, train_loaders, val_loaders, optimizer, loss_fn, device):
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
        total_steps = 0  # Add step counter
        for train_loader in train_loaders:
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
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_loss += averaged_loss.item()
                total_steps += 1  # Increment step counter
            
        # Average the loss by total number of steps
        train_loss = train_loss / total_steps  # Add this line
        args.writer.add_scalar('Loss/Train', train_loss, epoch)
        print(f"Epoch {epoch}/{args.epochs}, Training mean Loss: {train_loss}")
        
        if epoch % args.plot_interval == 0:
            # Evaluate on validation set
            val_loss = evaluate_valid_loss(model, val_loaders, loss_fn, device)
            args.writer.add_scalar('Loss/Val', val_loss, epoch)
            print(f"Epoch {epoch}/{args.epochs}, Validation mean Loss: {val_loss}")
            
        if epoch % args.plot_interval == 0:    
            for i in range(len(val_loaders)):
                # sample path visualization
                current_goal_distance, _map, trajectories, goal_positions = path_formation(args, model, val_loaders[i], 0, 0, device, action_choice="sample")
                animate_paths(args, i, epoch, trajectories, goal_positions, _map, interval=500)
                args.writer.add_scalar(f'Loss/video_goal_dis_{i}', current_goal_distance, epoch)
                print("current_goal_distance: ", current_goal_distance)

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

    args.writer.add_text('Args', args_str, 0)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)  # Set seed for torch
    np.random.seed(args.seed)     # Set seed for numpy
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)  # If using GPU, set the seed for all devices

    
    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=args.feature_dim, n_classes=args.action_dim, bilinear=False)
    
    # 计算可训练参数的总数
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # 计算模型的内存需求
    # 假设每个参数为32位浮点数（4字节）
    model_memory = total_params * 4 / (1024 ** 2)  # 转换为MB

    print(f"参数总数 (parameter):{total_params}")
    print(f"模型大小约为 (model size):{model_memory:.2f} MB")
    
    optimizer = torch.optim.AdamW(
                    net.parameters(),
                    lr=args.lr,
                    betas=(0.9, 0.999),  # 默认值，适合大多数情况
                    weight_decay=args.weight_decay
                )
    loss_fn = nn.CrossEntropyLoss(reduction="none")  
    
    # dataset 
    
    train_loaders = []
    val_loaders = []
    for dataset_path in args.dataset_paths:

        h5_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".h5")]

        test_list = random.sample(h5_files, int(0.1 * len(h5_files))) # 90% training, 10% validation
        train_list = [item for item in h5_files if item not in test_list]
        print("train_list size", len(train_list))
        print("test_list size", len(test_list))
        
        train_data = MAPFDataset(train_list, args.feature_dim) 
        test_data = MAPFDataset(test_list, args.feature_dim)
        train_loader = DataLoader(train_data, shuffle=True,  
                                batch_size=args.batch_size,  
                                num_workers=16)
        val_loader = DataLoader(test_data, shuffle=False,  
                                batch_size=args.batch_size, 
                                num_workers=16)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    # train
    train(args, net, train_loaders, val_loaders, optimizer, loss_fn, device)
