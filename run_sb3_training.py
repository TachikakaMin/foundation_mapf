#!/usr/bin/env python3
"""
便捷的SB3训练启动脚本
支持不同的训练配置模板
"""

import os
import sys
import subprocess


def run_training(config="simple"):
    """运行SB3训练"""
    
    configs = {
        "simple": {
            "script": "train_rl_sb3_simple.py",
            "args": [
                "--epochs", "50",
                "--num_agents", "32", 
                "--steps_per_epoch", "1024",
                "--mini_batch_size", "32"
            ]
        },
        "fast": {
            "script": "train_rl_sb3_simple.py",
            "args": [
                "--epochs", "20",
                "--num_agents", "16",
                "--steps_per_epoch", "512", 
                "--mini_batch_size", "16"
            ]
        },
        "production": {
            "script": "train_rl_sb3_simple.py",
            "args": [
                "--epochs", "100",
                "--num_agents", "64",
                "--steps_per_epoch", "2048",
                "--parallel_collect",
                "--num_workers_collect", "4"
            ]
        },
        "cpu": {
            "script": "train_rl_sb3_simple.py", 
            "args": [
                "--epochs", "30",
                "--num_agents", "24",
                "--steps_per_epoch", "1024",
                "--mini_batch_size", "16",
                "--force_cpu"
            ]
        }
    }
    
    if config not in configs:
        print(f"错误: 未知配置 '{config}'")
        print(f"可用配置: {', '.join(configs.keys())}")
        return False
    
    cfg = configs[config]
    cmd = ["python", cfg["script"]] + cfg["args"]
    
    print(f"=== 启动{config}配置训练 ===")
    print(f"命令: {' '.join(cmd)}")
    print("=" * 40)
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ {config}配置训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
        return False


def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        "stable_baselines3",
        "gymnasium", 
        "torch"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ 缺少依赖包:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n请运行: pip install -r requirements_sb3.txt")
        return False
    
    print("✅ 所有依赖已安装")
    return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SB3 MAPF训练启动器")
    parser.add_argument(
        "--config", "-c",
        choices=["simple", "fast", "production", "cpu"],
        default="simple",
        help="训练配置模板"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="检查依赖是否安装"
    )
    
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
        return
    
    print("🚀 MAPF强化学习训练 - Stable Baselines3版本")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 运行训练
    success = run_training(args.config)
    
    if success:
        print("\n🎉 训练任务完成!")
        print("📊 查看训练日志: tensorboard --logdir logs/")
    else:
        print("\n💥 训练失败，请检查错误信息")


if __name__ == "__main__":
    main()





