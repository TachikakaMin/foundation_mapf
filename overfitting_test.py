import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from args import get_args
from models.unet import UNet
from data_preprocess.datasetMAPF import MAPFDataset
from evaluation import evaluate_valid_loss
from path_visualization import path_formation, animate_paths

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
            averaged_loss = loss.mean() # scalar 
           
            
            # Backward pass and optimization
            optimizer.zero_grad()
            averaged_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            # Train loss for epoch
            train_loss += loss.sum().item()
        print(f"Epoch {epoch}/{args.epochs}, Training mean Loss: {train_loss}")

        # Evaluate on validation set
        val_loss = evaluate_valid_loss(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs}, Validation mean Loss: {val_loss}")
        
        # sample path visualization
        # if epoch % 5 == 0:
        if epoch  == 99:
            current_goal_distance, _map, trajectories, goal_positions = path_formation(model, val_loader, 0, 0, device)
            animate_paths(trajectories, goal_positions, _map, interval=500)
        
    return current_goal_distance


if __name__ == "__main__":
    seed = 1121
    torch.manual_seed(seed)  # Set seed for torch
    np.random.seed(seed)     # Set seed for numpy
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # If using GPU, set the seed for all devices

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
    # dataloaders
    train_loader = DataLoader(data, shuffle=True,  
                              batch_size=1,  
                              num_workers=0)
    val_loader = DataLoader(data, shuffle=False,  
                              batch_size=1, 
                              num_workers=0) 
    
    # train
    train(args, net, train_loader, val_loader, optimizer, loss_fn, device)