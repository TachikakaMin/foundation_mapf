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
        default=["data/input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-16.mbin"
                 ],
        help="sample data path",
    )
    parser.add_argument(
        "--num_workers", "-nw", type=int, default=20, help="number of workers"
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
    parser.add_argument("--first_layer_channels", "-flc", type=int, default=64, help="First layer channels")
    # 添加分布式训练参数
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file")
    
    return parser.parse_args()


def get_rl_args():
    """用于解析强化学习训练的命令行参数"""
    parser = argparse.ArgumentParser(description="MAPF Reinforcement Learning Training")

    # 继承基础参数
    parser.add_argument("--seed", "-sd", type=int, default=1919180, help="random seed")
    parser.add_argument("--log_dir", "-ld", type=str, default="runs", help="tensorboard log directory")
    
    # 数据集参数（RL用不到，但保持兼容性）
    parser.add_argument("--dataset_path", "-dp", type=str, default="data/input_data", help="dataset path")
    parser.add_argument("--sample_data_path", "-sp", type=str, nargs="+", 
                       default=["data/input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-16.mbin"],
                       help="sample data path")
    parser.add_argument("--num_workers", "-nw", type=int, default=20, help="number of workers")
    
    # 基础训练参数
    parser.add_argument("--epochs", "-ep", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5, help="Base learning rate", dest="lr")
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--eval_interval", type=int, default=2, help="evaluation interval (epochs)")
    parser.add_argument("--save_interval", type=int, default=2, help="model save interval (epochs)")
    
    # 模型参数
    parser.add_argument("--feature_dim", "-fd", type=int, default=6, help="feature dimension")
    parser.add_argument("--feature_type", "-ft", type=str, default="gradient", help="feature type")
    parser.add_argument("--steps", "-st", type=int, default=300, help="evaluation steps")
    parser.add_argument("--action_dim", "-ad", type=int, default=5, help="number of action types")
    parser.add_argument("--model", "-m", type=str, default="unet", help="model architecture type")
    parser.add_argument("--bilinear", action="store_true", default=False, help="use bilinear upsampling")
    parser.add_argument("--first_layer_channels", "-flc", type=int, default=64, help="first layer channels")
    parser.add_argument("--distributed", action="store_true", help="enable distributed training")
    parser.add_argument("--model_path", type=str, default=None, help="path to pretrained model file")
    
    # ============ 强化学习特有参数 ============
    # PPO算法参数
    parser.add_argument("--pi_lr", type=float, default=1e-4, help="策略网络学习率")
    parser.add_argument("--vf_lr", type=float, default=3e-4, help="价值网络学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="奖励折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO裁剪比率")
    parser.add_argument("--train_pi_iters", type=int, default=10, help="策略网络更新次数")
    parser.add_argument("--train_v_iters", type=int, default=10, help="价值网络更新次数")
    parser.add_argument("--target_kl", type=float, default=0.1, help="目标KL散度（早停阈值）")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵正则化系数")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="PPO训练时的mini-batch大小")
    
    # RL训练流程参数
    parser.add_argument("--steps_per_epoch", type=int, default=800, help="每个epoch收集的环境步数")
    parser.add_argument("--num_eval_episodes", type=int, default=10, help="策略评估回合数")
    
    # MAPF环境参数
    parser.add_argument("--num_agents", type=int, default=64, help="环境中的智能体数量")
    parser.add_argument("--max_episode_steps", type=int, default=200, help="单个回合最大步数")
    
    # 多线程收集参数
    parser.add_argument("--parallel_collect", action="store_true", default=False, help="是否使用多线程并行经验收集")
    parser.add_argument("--num_workers_collect", type=int, default=16, help="经验收集工作线程数")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
