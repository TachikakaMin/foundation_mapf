import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from args import get_args
from models.unet import UNet
from dataset.datasetMAPF import MAPFDataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(args, model, train_loader):
    optimizer = torch.optim.RMSprop(model.parameters(),
                              lr=args.lr, weight_decay=1e-5, momentum=0.999, foreach=True)
    criterion = nn.CrossEntropyLoss(reduction="none")
    global_step = 0
    model.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            feature = batch["feature"].to(device)
            action_y = batch["action"].to(device)
            mask = batch["mask"].to(device)
            pred = model(feature)
            loss = criterion(pred, action_y)
            loss = (loss * mask.float()).max()
            # non_zero_elements = mask.sum()
            # loss = loss/non_zero_elements
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            global_step += 1
            epoch_loss += loss.item()
        print(epoch_loss)

if __name__ == "__main__":
    args = get_args() 
    
    # the number of binary bits can be used to represent the number of agents
    args.feature_dim = int(np.ceil(np.log2(args.max_agent_num)))
    
    # every grid of a input map is represented by a feature vector with feature_dim*2+1 features
    # the first feature represents the existence of obstacle
    # the next feature_dim features represents the goal position of a specific agent
    # the next feature_dim features represents the start position of a specific agent
    net = UNet(args.feature_dim*2+1, args.action_dim)  
    
    train_data = MAPFDataset(args.dataset_path, args.feature_dim)
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=args.batch_size, 
                              num_workers=1)
    
    train(args, net, train_loader)
    
    
    
