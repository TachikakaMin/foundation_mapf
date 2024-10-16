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
from tqdm import tqdm

device = "cuda"

def generate_test(seed=0):
    np.random.seed(seed=seed)
    h, w = 16, 16
    mp_feature = np.zeros((h,w, 1), dtype=int)
    agent_num = 10
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
    
    return agent_num, agent_cur, agent_goals, goal_loc_info, mp_feature
    


def move_agent(cur_loc, act, mp_feature):
    new_loc = dp(cur_loc)
    act = act.detach().cpu().numpy()
    agent_num = new_loc.shape[0]
    h, w = mp_feature.shape[:2]
    for i in range(agent_num):
        loc = new_loc[i]
        action_idx = act[loc[0], loc[1]]
        if action_idx == 0: # left
            loc[1] = max(loc[1]-1, 0)
        if action_idx == 1: # right
            loc[1] = min(loc[1]+1, w-1)
        if action_idx == 2: # up
            loc[0] = max(loc[0]-1, 0)
        if action_idx == 3: # down
            loc[0] = min(loc[0]+1, h-1)
        new_loc[i] = loc
    while True:
        b = 1
        occupy_map = 2*mp_feature
        for i in range(agent_num):
            loc = new_loc[i]
            occupy_map[loc[0], loc[1]] += 1
        for i in range(agent_num):
            loc = new_loc[i]
            if occupy_map[loc[0], loc[1]] > 1:
                new_loc[i] = cur_loc[i]
                b = 0
        if b: break
    
    return new_loc

def cal_dis(agent_cur, agent_goals):
    agent_num = len(agent_cur)
    ans = 0
    for i in range(agent_num):
        ans += abs(agent_cur[i][0]-agent_goals[i][0])
        ans += abs(agent_cur[i][1]-agent_goals[i][1])
    return ans

def test(args, model):
    model.eval()
    agent_num, agent_cur, agent_goals, goal_loc_info, mp_feature = generate_test()
    h, w = mp_feature.shape[:2]
    for i in range(100):
        current_loc_info = np.zeros((h, w, args.agent_dim), dtype=int)
        indices = np.arange(agent_num)
        binary_strings = np.array([list(format(i+1, f'0{args.agent_dim}b')) for i in indices], dtype=int)
        current_loc_info[agent_cur[:, 0], agent_cur[:, 1]] = binary_strings
        current_loc_info = torch.FloatTensor(current_loc_info)
        
        in_feature = [mp_feature, goal_loc_info, current_loc_info]
        in_feature = torch.cat(in_feature, dim=-1).permute((2, 0, 1))
        in_feature = in_feature.unsqueeze(0).to(device)
        with torch.no_grad():
            _, out_feature = model(in_feature)
        out_feature = out_feature.squeeze(0).permute((1, 2, 0)).argmax(-1)
        mask = current_loc_info.any(-1).to(device)
        action = out_feature * mask
        agent_cur = move_agent(agent_cur, action, mp_feature)
        not_done = cal_dis(agent_cur, agent_goals)
        if not not_done:
            return 0
    cost2 = cal_dis(agent_cur, agent_goals)
    return cost2

def train(args, model, train_loader):
    optimizer = torch.optim.RMSprop(model.parameters(),
                              lr=args.lr, weight_decay=1e-5, momentum=0.999, foreach=True)
    criterion = nn.CrossEntropyLoss(reduction="none")  #  reduction="none"，因此每个样本的损失都保留为独立值(不进行取平均之类的操作。结合掩码（mask）来只计算某些特定样本的损失，或 reduction="none"，因此每个样本的损失都保留为独立值。这个设计通常用于后续进行某些自定义操作，比如结合掩码（mask）来只计算某些特定样本的损失。
    global_step = 0
    model.to(device)
    model.train()
    for epoch in range(1, args.epochs + 1):
        
        epoch_loss = 0
        for batch in tqdm(train_loader):
            feature = batch["feature"].to(device)  # dimension = batch_size
            action_y = batch["action"].to(device)
            mask = batch["mask"].to(device)
            pred = model(feature)
            loss = criterion(pred, action_y)  # 返回的是一个与 pred 和 action_y 形状相同的张量，每个元素对应于某个样本的某个位置的分类损失。
            loss = (loss * mask.float()).max()  # 只取loss最大的某个样本中的某个位置  # 为啥不取平均值
            # non_zero_elements = mask.sum()
            # loss = loss/non_zero_elements
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # 防止梯度爆炸，通过对模型参数的梯度进行裁剪，限制其范数的最大值不超过 1。
            optimizer.step()
            global_step += 1
            epoch_loss += loss.item()
        print(epoch_loss)

if __name__ == "__main__":
    args = get_args() 
    args.agent_dim = int(np.ceil(np.log2(args.max_agent_num)))
    
    net = UNet(args.agent_dim*2+1, args.action_dim)
    
    
    train_data = MAPFDataset(args.dataset_path, args.feature_dim)  # input shape: (num_samples, 21, 32, 32) action output shape: (num_samples, 32, 32)
    train_loader = DataLoader(train_data, shuffle=True,  
                              batch_size=args.batch_size, 
                              num_workers=1)
    
    train(args, net, train_loader)