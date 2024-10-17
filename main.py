import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy as dp
from args import get_args
from models.unet import UNet
from dataset.datasetMAPF import MAPFDataset
from evaluation import evaluate
from tqdm import tqdm

def train(args, model, train_loader, optimizer, loss_fn, device='cuda'):
    """
    Trains the UNet model using masked loss, gradient clipping, and custom optimizer.

    Args:
        args: Argument object that contains training configurations like learning rate and epochs.
        model: The neural network model (UNet).
        train_loader: Dataloader for the training dataset.
        optimizer: Optimizer for training (optional).
        loss_fn: Loss function (optional, default is CrossEntropyLoss with reduction="none").
        device: Device to run the training on (default is 'cuda').
    """
    
    model.to(device)
    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0

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
            max_loss = averaged_loss.max()  #Select the maximum value from the [32, 32] averaged loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            max_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            epoch_loss += loss.item()
        test_loss = evaluate(args, model)
        model.train()
        print(epoch_loss, test_loss)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {epoch_loss}")


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
    # net.use_checkpointing() 
    optimizer = torch.optim.RMSprop(net.parameters(),
                              lr=args.lr, weight_decay=1e-8, momentum=0.999, foreach=True)
    loss_fn = nn.CrossEntropyLoss(reduction="none")  #  reduction="none"，因此每个样本的损失都保留为独立值(不进行取平均之类的操作。结合掩码（mask）来只计算某些特定样本的损失，或 reduction="none"，因此每个样本的损失都保留为独立值。这个设计通常用于后续进行某些自定义操作，比如结合掩码（mask）来只计算某些特定样本的损失。

    
    # dataset 
    train_data = MAPFDataset(args.dataset_path, args.agent_dim)  # input shape: (num_samples, 21, 32, 32) action output shape: (num_samples, 32, 32)
    train_loader = DataLoader(train_data, shuffle=True,  
                              batch_size=args.batch_size, 
                              num_workers=1)
    
    # train
    train(args, net, train_loader)
    
    ### 到时候数据需要去掉所有重复的
    