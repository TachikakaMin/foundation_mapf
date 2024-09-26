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

device = "cuda"

def move_agent(cur_loc, act, h, w):
    agent_num = cur_loc.shape[0]
    for i in range(agent_num):
        loc = cur_loc[i]
        action_idx = act[loc[0], loc[1]]
        if action_idx == 0:
            loc[0] = a

def test(args, model):
    model.eval()
    h, w = 16, 16
    mp_feature = np.zeros((h,w, 1), dtype=int)
    agent_num = np.random.randint(1, 10)
    all_positions = [[x, y] for x in range(h) for y in range(w)]
    np.random.shuffle(all_positions)
    agent_cur = np.array(all_positions[:agent_num])
    agent_goals = np.array(all_positions[agent_num:2 * agent_num])
    
    goal_loc_info = np.zeros((h, w, args.agent_dim), dtype=int)
    indices = np.arange(agent_num)
    binary_strings = np.array([list(format(i+1, f'0{args.agent_dim}b')) for i in indices], dtype=int)
    goal_loc_info[agent_goals[:, 0], agent_goals[:, 1]] = binary_strings
    goal_loc_info = torch.FloatTensor(goal_loc_info)
    mp_feature = torch.FloatTensor(mp_feature)
    for i in range(1000):
        current_loc_info = np.zeros((h, w, args.agent_dim), dtype=int)
        indices = np.arange(agent_num)
        binary_strings = np.array([list(format(i+1, f'0{args.agent_dim}b')) for i in indices], dtype=int)
        current_loc_info[agent_cur[:, 0], agent_cur[:, 1]] = binary_strings
        current_loc_info = torch.FloatTensor(current_loc_info)
        
        in_feature = [mp_feature, goal_loc_info, current_loc_info]
        in_feature = torch.cat(in_feature, dim=-1).permute((2, 0, 1))
        in_feature = in_feature.unsqueeze(0).to(device)
        with torch.no_grad():
            out_feature = model(in_feature)
        out_feature = out_feature.squeeze(0).permute((1, 2, 0)).argmax(-1)
        mask = current_loc_info.any(-1).to(device)
        action = out_feature * mask
        agent_cur = move_agent(agent_cur, action)
        
            
    return mp_feature

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
            # test(args, model)
        print(epoch_loss)

if __name__ == "__main__":
    args = get_args() 
    args.agent_dim = int(np.ceil(np.log2(args.max_agent_num)))
    
    net = UNet(args.agent_dim*2+1, args.action_dim)
    
    train_data = MAPFDataset(args.dataset_path, args.agent_dim)
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=args.batch_size, 
                              num_workers=1)
    
    train(args, net, train_loader)
    
    
    
