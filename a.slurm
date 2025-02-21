#!/bin/bash
#SBATCH --partition=ghx4
#SBATCH --account=bcjw-dtai-gh
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=500G
#SBATCH --time=20:00:00

torchrun --nproc_per_node=4 train.py --distributed -nw 64 -ep 10 -fd 5
torchrun --nproc_per_node=4 train.py --distributed -nw 64 -ep 10 -fd 6
torchrun --nproc_per_node=4 train.py --distributed -nw 64 -ep 10 -m cnn
