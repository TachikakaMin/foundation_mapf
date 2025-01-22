import argparse


def get_args():
    """用于解析命令行参数"""
    parser = argparse.ArgumentParser(description="UNet training")

    parser.add_argument("--seed", "-sd", type=int, default=1919180, help="seed")
    parser.add_argument("--log_dir", "-ld", type=str, default="runs", help="plot log")

    # dataset
    parser.add_argument(
        "--dataset_path",
        "-dp",
        type=str,
        default="data/input_data",
        help="dataset path",
    )
    parser.add_argument(
        "--sample_data_path",
        "-sp",
        type=str,
        nargs="+",
        default=["data/input_data/maze-32-32-20-0/maze-32-32-20-0-64/maze-32-32-20-0-64-0/maze-32-32-20-0-64-0-0.bin",
                 "data/input_data/maze-32-32-20-1/maze-32-32-20-1-64/maze-32-32-20-1-64-0/maze-32-32-20-1-64-0-0.bin",
                 "data/input_data/random-32-32-20-1/random-32-32-20-1-64/random-32-32-20-1-64-0/random-32-32-20-1-64-0-0.bin",
                 "data/input_data/random-32-32-20-2/random-32-32-20-2-64/random-32-32-20-2-64-0/random-32-32-20-2-64-0-0.bin",
                 ],
        help="sample data path",
    )
    parser.add_argument(
        "--num_workers", "-nw", type=int, default=128, help="number of workers"
    )
    # training
    parser.add_argument(
        "--epochs", "-ep", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--weight_decay", "-wd", type=float, default=1e-3, help="weight decay"
    )
    parser.add_argument("--eval_interval", type=int, default=2, help="eval interval")
    parser.add_argument("--save_interval", type=int, default=2, help="save interval")
    # model
    parser.add_argument(
        "--feature_dim", "-fd", type=int, default=6, help="feature dimension"
    )
    parser.add_argument("--feature_type", "-ft", type=str, default="gradient", help="feature type")
    parser.add_argument("--steps", "-st", type=int, default=300, help="Steps")
    parser.add_argument("--action_dim", "-ad", type=int, default=5, help="Action types")
    parser.add_argument("--model", "-m", type=str, default="unet", help="Model type")
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    # 添加分布式训练参数
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
