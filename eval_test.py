import torch
from models.unet import UNet
from MAPF_dataset import MAPFDataset
from torch.utils.data import DataLoader
from tools.path_formation import path_formation
from tools.visualize_path import visualize_path
import numpy as np
import random
import argparse


def main():
    # 获取参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the dataset files",
    )
    parser.add_argument("--feature_dim", type=int, default=6, help="Feature dimension")
    parser.add_argument("--feature_type", type=str, default="gradient", help="Feature type")
    parser.add_argument("--steps", type=int, default=300, help="Steps")
    parser.add_argument("--action_dim", type=int, default=5, help="Action dimension")
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evals",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show the path",
    )
    args = parser.parse_args()

    # 设置随机种子以确保可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = UNet(
        n_channels=args.feature_dim, n_classes=args.action_dim, bilinear=args.bilinear
    ).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 准备验证数据集
    test_dataset = MAPFDataset(args.dataset_paths, args.feature_dim, args.feature_type)
    val_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    all_paths, goal_locations, _, file_name = path_formation(
        model, val_loader, 0, device, args.feature_type, steps=args.steps
    )
    visualize_path(all_paths, goal_locations, file_name, video_path=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
