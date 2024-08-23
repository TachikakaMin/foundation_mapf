import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from args import get_args
from models.unet import UNet
from dataset.datasetMAPF import MAPFDataset


if __name__ == "__main__":
    args = get_args() 
    args.feature_dim = int(np.ceil(np.log2(args.max_agent_num))) # 确保用二进制表示智能体的数量
    
    # 模型初始化
    net = UNet(args.feature_dim, args.action_dim)
    
    
    train_data = MAPFDataset("data/sample.yaml", args.feature_dim)
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=args.batch_size, 
                              num_workers=os.cpu_count())
    
    optimizer = optim.RMSprop(net.parameters(),
                              lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, foreach=True)
    # 在某个指标（例如Dice得分）停止提高时降低学习率。它配置为在5个epoch（轮次）内没有观察到改进时减少学习率。
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 
    # 自动缩放梯度，防止在混合精度训练中出现梯度下溢
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss()
    #global_step = 0