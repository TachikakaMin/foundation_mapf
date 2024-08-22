import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from args import get_args
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from models.unet import UNet
from dataset.datasetMAPF import MAPFDataset


if __name__ == "__main__":
    args = get_args()
    args.feature_dim = int(np.ceil(np.log2(args.max_agent_num)))
    net = UNet(args.feature_dim, args.action_dim)
    train_data = MAPFDataset("data/sample.yaml", args.feature_dim)
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=args.batch_size, 
                              num_workers=os.cpu_count())
    
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0