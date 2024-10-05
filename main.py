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
        model.eval()
        batch = train_loader.dataset[0]
        for i in range(47):
            feature = batch["feature"].unsqueeze(0).to(device)
            mask = batch["mask"].unsqueeze(0).to(device)
            pred = model(feature)
            pred = pred.argmax(dim=1) * mask
            new_feature = feature.clone()
            robot_locations = mask[0].nonzero()

            directions = {
                1: (-1, 0),  # Move up
                2: (0, 1),   # Move right
                3: (1, 0),   # Move down
                4: (0, -1)   # Move left
            }
            # Iterate over all robot locations
            for loc in robot_locations:
                i, j = loc  # Get the (i, j) location of the robot

                # Get the action from the pred tensor for this location
                action = pred[i, j]

                # Get the movement direction based on the action
                if action in directions:  # Ignore if action is 0 (no movement)
                    di, dj = directions[action]
                    
                    # Compute the new location
                    new_i, new_j = i + di, j + dj

                    # Ensure the new location is within bounds
                    if 0 <= new_i < 32 and 0 <= new_j < 32:
                        # Move the robot to the new location by updating the last 10 dimensions
                        # Clear robot's presence from the current location
                        new_feature[11:, i, j] = 0

                        # Set robot's presence in the new location
                        new_feature[11:, new_i, new_j] = feature[11:, i, j]
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