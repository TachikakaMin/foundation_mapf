import os
import torch
from models.unet import UNet
from data_preprocess.datasetMAPF import MAPFDataset
from torch.utils.data import DataLoader
from evaluation import evaluate_valid_loss
from path_visualization import path_formation, animate_paths
from args import get_args
import numpy as np
import random
from datetime import datetime

def main():
    # Get the same arguments as used in training
    args = get_args()
    args.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    feature_channels = args.feature_dim
    net = UNet(n_channels=feature_channels, n_classes=args.action_dim, bilinear=False)
    
    # Load checkpoint
    net.load_state_dict(torch.load(args.load))
    net.to(device)
    net.eval()
    
    val_loaders = []
    for dataset_path in args.dataset_paths:
        h5_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                   if f.endswith(".h5")]
        test_files = h5_files[:10]
        
        test_data = MAPFDataset(test_files, args.feature_dim)
        val_loader = DataLoader(test_data, 
                              shuffle=False,  
                              batch_size=args.batch_size, 
                              num_workers=16)
        val_loaders.append(val_loader)
    
    # Set up loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    # Evaluate
    with torch.no_grad():
        # Calculate validation loss
        val_loss = evaluate_valid_loss(net, val_loaders, loss_fn, device)
        print(f"Validation Loss: {val_loss}")
        
        # Generate and animate paths for each validation loader
        for i, val_loader in enumerate(val_loaders):
            print(f"Generating path visualization for map type {args.dataset_paths[i]}")
            current_goal_distance, _map, trajectories, goal_positions = path_formation(
                args, net, val_loader, 0, 0, device, action_choice="sample"
            )
            print(f"Goal distance: {current_goal_distance}")
            
            # Save animation
            animate_paths(args, i, 10, trajectories, goal_positions, _map, interval=500)

if __name__ == "__main__":
    main()
