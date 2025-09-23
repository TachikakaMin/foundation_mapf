import argparse
import os


def get_sb3_args():
    """获取Stable Baselines3强化学习训练参数"""
    parser = argparse.ArgumentParser(description="MAPF RL Training with Stable Baselines3")
    
    # 基础训练参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志和模型保存目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU训练")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "cnn"], help="模型架构类型")
    parser.add_argument("--first_layer_channels", type=int, default=64, help="第一层通道数")
    parser.add_argument("--bilinear", action="store_true", help="是否使用双线性插值")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型路径")
    
    # PPO算法参数
    parser.add_argument("--pi_lr", type=float, default=3e-4, help="策略网络学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO裁剪比率")
    parser.add_argument("--train_pi_iters", type=int, default=10, help="每次更新的优化步数")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵正则化系数")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Mini-batch大小")
    
    # 环境参数
    parser.add_argument("--steps_per_epoch", type=int, default=2048, help="每轮收集的步数")
    parser.add_argument("--num_agents", type=int, default=64, help="智能体数量")
    parser.add_argument("--max_episode_steps", type=int, default=200, help="单个回合最大步数")
    parser.add_argument("--feature_dim", type=int, default=6, help="特征维度")
    parser.add_argument("--feature_type", type=str, default="gradient", help="特征类型")
    parser.add_argument("--action_dim", type=int, default=5, help="动作维度")
    
    # 评估和保存参数
    parser.add_argument("--eval_interval", type=int, default=10, help="评估间隔(epochs)")
    parser.add_argument("--save_interval", type=int, default=20, help="模型保存间隔(epochs)")
    parser.add_argument("--num_eval_episodes", type=int, default=10, help="评估回合数")
    
    # 并行参数
    parser.add_argument("--parallel_collect", action="store_true", help="是否使用多进程收集经验")
    parser.add_argument("--num_workers_collect", type=int, default=4, help="经验收集进程数")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_sb3_args()
    print("=== SB3 Training Arguments ===")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=============================")





