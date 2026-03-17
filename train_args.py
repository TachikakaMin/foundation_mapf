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
        help="offline dataset path (.mbin files)",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default=None,
        help="offline validation dataset path (.mbin files), defaults to --dataset_path",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        choices=["offline", "online"],
        default="offline",
        help="training dataset mode: offline=.mbin, online=on-the-fly generation",
    )
    parser.add_argument(
        "--train_map_path",
        type=str,
        default="data/map_files",
        help="map directory for online training mode",
    )
    parser.add_argument(
        "--online_total_steps",
        type=int,
        default=200000,
        help="total optimizer steps in online mode",
    )
    parser.add_argument(
        "--online_eval_interval_steps",
        type=int,
        default=4000,
        help="run validation every N optimizer steps in online mode",
    )
    parser.add_argument(
        "--online_save_interval_steps",
        type=int,
        default=4000,
        help="save checkpoint every N optimizer steps in online mode",
    )
    parser.add_argument(
        "--online_inference_test_interval_steps",
        type=int,
        default=4000,
        help="run inference test every N optimizer steps in online mode",
    )
    parser.add_argument(
        "--online_time_limit_sec",
        type=int,
        default=5,
        help="time limit for each online scenario generation",
    )
    parser.add_argument(
        "--online_retry_limit",
        type=int,
        default=20,
        help="max retries for online scenario generation",
    )
    parser.add_argument(
        "--sample_data_path",
        "-sp",
        type=str,
        nargs="+",
        default=["data/input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-16.mbin"
                 ],
        help="fixed offline sample .mbin files used by inference test during training",
    )
    parser.add_argument(
        "--inference_num_cases",
        type=int,
        default=1,
        help="number of fixed sample cases to run for inference test",
    )
    parser.add_argument(
        "--inference_test_interval",
        type=int,
        default=0,
        help="offline mode only: run inference test every N epochs; 0 means reuse --eval_interval",
    )
    parser.add_argument(
        "--inference_action_choice",
        type=str,
        choices=["sample", "max"],
        default="max",
        help="action selection used during inference test rollout",
    )
    parser.add_argument(
        "--num_workers", "-nw", type=int, default=20, help="number of workers"
    )
    # training
    parser.add_argument(
        "--epochs", "-ep", type=int, default=100, help="offline mode only: number of epochs"
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
    parser.add_argument("--eval_interval", type=int, default=2, help="offline mode only: eval interval in epochs")
    parser.add_argument("--save_interval", type=int, default=2, help="offline mode only: save interval in epochs")
    # model
    parser.add_argument(
        "--feature_dim", "-fd", type=int, default=6, help="feature dimension"
    )
    parser.add_argument("--feature_type", "-ft", type=str, default="gradient", help="feature type")
    parser.add_argument(
        "--steps",
        "-st",
        type=int,
        default=300,
        help="maximum rollout steps for inference test / path_formation",
    )
    parser.add_argument("--action_dim", "-ad", type=int, default=5, help="Action types")
    parser.add_argument("--model", "-m", type=str, default="unet", help="Model type")
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument("--first_layer_channels", "-flc", type=int, default=64, help="First layer channels")
    # 添加分布式训练参数
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
