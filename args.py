import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    # training
    parser.add_argument('--epochs', '-ep', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-bs', dest='batch_size', metavar='B', 
                        type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-l', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')


    #MAPF settings
    parser.add_argument('--max_agent_num', '-na', metavar='NA', type=int, default=1000, 
                        help='Max number of agents')
    parser.add_argument('--action_dim', '-ad', metavar='AD', type=int, default=5, help='Action types')
    return parser.parse_args()