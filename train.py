import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from train_args import get_args
from models.unet import UNet
from models.CNN import CNN
from MAPF_dataset import MAPFDataset
from tools.path_formation import path_formation
from torch.utils.tensorboard import SummaryWriter
import random
import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import psutil  # Add this import


def evaluate_valid_loss(model, val_loader, loss_fn, device):
    # Set model to evaluation mode
    model.eval()
    val_loss = 0
    total_agents = 0  # Add step counter

    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(val_loader, desc="Evaluating", disable=args.local_rank!=0):
            # Load validation data onto the correct device (CPU/GPU)
            feature = batch["feature"].to(device)
            action_y = batch["action"].to(device)
            mask = batch["mask"].to(device)
            # Forward pass
            logits, _ = model(feature)

            # Compute the loss and apply mask
            loss = loss_fn(logits, action_y)
            masked_loss = loss * mask.float()
            val_loss += masked_loss.detach().sum().item()
            total_agents += mask.detach().sum().item()  # Increment agent counter

    return val_loss / total_agents  # Average the loss by total agents

def train(args, model, train_loaders, val_loaders, sample_loader, optimizer, loss_fn, device):
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        total_agents = 0
        for train_loader in train_loaders.values():
            for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}/{args.epochs}", disable=args.local_rank!=0, total=len(train_loader)):
                # # Add per-batch memory print
                # if i % 100 == 0:  # Print every 100 batches
                #     torch.cuda.empty_cache()
        
                #     # Force garbage collection
                #     import gc
                #     gc.collect()
                #     total_memory = 0
                #     main_process = psutil.Process()
                #     processes = [main_process] + main_process.children(recursive=True)
                #     print(f"\nBatch {i} memory usage:")
                #     print(f"Number of processes: {len(processes)}")
                    
                #     for proc in processes:
                #         try:
                #             memory_info = proc.memory_info()
                #             total_memory += memory_info.rss
                #         except (psutil.NoSuchProcess, psutil.AccessDenied):
                #             continue
                    
                #     print(f"Total Memory Used by All Processes: {total_memory / 1024 / 1024:.2f} MB")
                #     system_memory = psutil.virtual_memory()
                #     print(f"System Memory Usage: {system_memory.percent}%")
                # Load data onto the correct device (CPU/GPU)
                feature = batch["feature"].to(
                    device
                )  # shape:[batch_size, channel_num, n, m]
                action_y = batch["action"].to(device)  # shape:[batch_size, n, m]
                mask = batch["mask"].to(device)  # shape:[batch_size, n, m]

                # Forward pass
                logit, _ = model(feature)  # shape:[batch_size, action_dim, n, m]

                # Compute loss and apply mask
                loss = loss_fn(logit, action_y)  # shape:[batch_size, n, m]
                masked_loss = loss * mask.float()
                averaged_loss = masked_loss.sum() / mask.sum()  # scalar
                
                # Backward pass and optimization
                optimizer.zero_grad()
                averaged_loss.backward()
                
                # Update accumulated loss stats
                train_loss += masked_loss.detach().sum().item()
                total_agents += mask.detach().sum().item()

                # Gradient clipping to prevent exploding gradients
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

        # Average the loss by total number of steps
        train_loss = train_loss / total_agents  # Add this line
        if args.local_rank == 0:
            args.writer.add_scalar("Loss/Train", train_loss, epoch)
            print(f"Epoch {epoch}/{args.epochs}, Training mean Loss: {train_loss}")
        if epoch % args.eval_interval == 0 and args.local_rank == 0:
            for val_loader in val_loaders.values():
                val_loss = evaluate_valid_loss(model, val_loader, loss_fn, device)
                args.writer.add_scalar("Loss/Val", val_loss, epoch)
                print(f"Epoch {epoch}/{args.epochs}, Validation mean Loss: {val_loss}")
            # sample path visualization
            for idx in range(len(sample_loader)):   
                _, _, current_goal_distance, _ = path_formation(model, sample_loader, idx, device, args.feature_type, steps=args.steps)
                args.writer.add_scalar(
                    f"Loss/video_goal_dis_{idx}", current_goal_distance, epoch
                )
        if epoch % args.save_interval == 0 and args.local_rank == 0:
            file_path = os.path.join(
                args.real_log_dir, f"model_checkpoint_epoch_{epoch}.pth"
            )
            if args.distributed:
                torch.save(model.module.state_dict(), file_path)
            else:
                torch.save(model.state_dict(), file_path)

if __name__ == "__main__":

    # arguments
    args = get_args()

    # 设置分布式训练
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        args.local_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda:{}".format(args.local_rank))
    else:
        args.local_rank = 0    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.real_log_dir = os.path.join(args.log_dir, f"{args.current_time}")
    
    # 只在主进程上创建tensorboard writer
    if args.local_rank == 0:
        args.writer = SummaryWriter(log_dir=args.real_log_dir)
        args_dict = vars(args)
        args_str = "\n".join([f"{key}: {value}" for key, value in args_dict.items()])
        args.writer.add_text("Args", args_str, 0)
        print(args_str)

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # model
    if args.model == "unet":
        net = UNet(n_channels=args.feature_dim, n_classes=args.action_dim, first_layer_channels=args.first_layer_channels, bilinear=args.bilinear)
    elif args.model == "cnn":
        net = CNN(n_channels=args.feature_dim, n_classes=args.action_dim)
    net.to(device)
    # 如果使用分布式训练，将模型包装为DDP模型
    if args.distributed:
        net = DDP(net, device_ids=[args.local_rank])

    # 只在主进程上打印模型信息
    # if args.local_rank == 0:
    #     total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #     model_memory = total_params * 4 / (1024**2)
    #     print(f"参数总数 (parameter):{total_params}")
    #     print(f"模型大小约为 (model size):{model_memory:.2f} MB")

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),  # 默认值，适合大多数情况
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # dataset
    train_loaders = {}  # Dictionary to store loaders by map dimensions
    val_loaders = {}
    
    def get_map_dims(filename):
        # Extract h,w from filename like 'maze-32-32-20-0-32-0-0.bin'
        parts = filename.split('-')
        return (int(parts[1]), int(parts[2]))


    # Get immediate subdirectories of dataset_path
    dimension_groups = {}
    subdirs = [d for d in os.listdir(args.dataset_path)]
    
    for subdir in subdirs:
        dims = get_map_dims(subdir)
        # Get all .bin files in this subdirectory recursively
        dir_path = os.path.join(args.dataset_path, subdir)
        files = glob.glob(os.path.join(dir_path, "**/*.bin"), recursive=True)
        if dims not in dimension_groups:
            dimension_groups[dims] = []
        dimension_groups[dims] += files
        if args.local_rank == 0:
            print(f"Found {len(files)} .bin files in {subdir}")

    # Create separate dataloaders for each dimension group
    for dims, files in dimension_groups.items():
        n_files = len(files)
        indices = list(range(n_files))
        
        if not args.distributed:
            random.shuffle(indices)
        
        n_test = int(0.1 * n_files)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        test_list = [files[i] for i in test_indices]
        train_list = [files[i] for i in train_indices]
        
        if args.local_rank == 0:
            print(f"Map size {dims} - train_list size: {len(train_list)}, test_list size: {len(test_list)}")

        train_data = MAPFDataset(train_list, args.feature_dim, args.feature_type)
        test_data = MAPFDataset(test_list, args.feature_dim, args.feature_type)
        
        # 为分布式训练添加采样器
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        val_sampler = torch.utils.data.distributed.DistributedSampler(test_data) if args.distributed else None

        train_loader = DataLoader(
            train_data,
            shuffle=(train_sampler is None),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_sampler
        )
        val_loader = DataLoader(
            test_data,
            shuffle=(val_sampler is None),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=val_sampler
        )

        train_loaders[dims] = train_loader
        val_loaders[dims] = val_loader
    
    sample_data = MAPFDataset(args.sample_data_path, args.feature_dim, args.feature_type, first_step=True)
    sample_loader = DataLoader(
        sample_data,
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )
    # train
    train(args, net, train_loaders, val_loaders, sample_loader, optimizer, loss_fn, device)
