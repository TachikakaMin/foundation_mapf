import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from args import get_args
from models.unet import UNet
from dataset.datasetMAPF import MAPFDataset
from tqdm import tqdm


def train(args, model, train_loader):
    optimizer = torch.optim.RMSprop(model.parameters(),
                              lr=args.lr, weight_decay=1e-8, momentum=0.999, foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    criterion = nn.CrossEntropyLoss(reduction="mean")
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            feature = batch["feature"]
            action_y = batch["action"]
            mask = batch["mask"]
            pred = model(feature)
            pred = torch.permute(pred, (0, 2, 3, 1))
            masked_pred = pred * mask.int().float()
            loss = criterion(masked_pred, action_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            global_step += 1
            epoch_loss += loss.item()
        print(epoch_loss)

if __name__ == "__main__":
    args = get_args() 
    args.feature_dim = int(np.ceil(np.log2(args.max_agent_num))) # 确保用二进制表示智能体的数量
    
    # 模型初始化
    net = UNet(args.feature_dim*2+1, args.action_dim)
    
    
    train_data = MAPFDataset("data/sample.yaml", args.feature_dim)
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=args.batch_size, 
                              num_workers=1)
    
    train(args, net, train_loader)
    
    
    
