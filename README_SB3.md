# MAPF强化学习训练 - Stable Baselines3版本

这个版本使用Stable Baselines3库进行MAPF（多智能体路径规划）的强化学习训练，相比自定义PPO实现更加稳定和易用。

## 安装依赖

```bash
pip install -r requirements_sb3.txt
```

## 训练模型

### 简化版本（推荐）
```bash
python train_rl_sb3_simple.py --epochs 50 --num_agents 32 --parallel_collect
```

### 完整版本
```bash
python train_rl_sb3.py --epochs 100 --num_agents 64
```

## 主要参数说明

- `--epochs`: 训练轮数 (默认: 100)
- `--num_agents`: 智能体数量 (默认: 64)  
- `--steps_per_epoch`: 每轮收集的步数 (默认: 2048)
- `--mini_batch_size`: Mini-batch大小 (默认: 64)
- `--pi_lr`: 学习率 (默认: 3e-4)
- `--parallel_collect`: 使用多进程收集经验
- `--num_workers_collect`: 工作进程数 (默认: 4)
- `--force_cpu`: 强制使用CPU训练

## 训练监控

- **TensorBoard**: 启动训练后，可以使用TensorBoard查看训练进度
  ```bash
  tensorboard --logdir logs/
  ```

- **模型保存**: 训练过程中会自动保存最佳模型和检查点到日志目录

## 版本说明

### train_rl_sb3_simple.py (推荐)
- **优点**: 代码简洁，训练稳定，显存需求低
- **动作空间**: 简化为离散选择，使用启发式动作映射
- **适用场景**: 快速实验，资源受限环境

### train_rl_sb3.py (完整版)
- **优点**: 保留原有模型结构，更接近原始问题
- **动作空间**: Box空间，直接处理空间动作
- **适用场景**: 需要精确控制动作空间的研究

## 与原版对比

| 特性 | 原版PPO | SB3版本 |
|------|---------|---------|
| **代码复杂度** | 高 (1000+行) | 低 (300-400行) |
| **显存需求** | 高 (~2GB+) | 低 (~1GB) |
| **训练稳定性** | 需要调试 | 开箱即用 |
| **功能完整性** | 完整 | 简化但实用 |
| **维护成本** | 高 | 低 |

## 故障排除

### 1. 显存不足
- 使用 `--force_cpu` 强制CPU训练
- 减少 `--mini_batch_size` 
- 减少 `--num_agents`

### 2. 训练不收敛
- 调整学习率 `--pi_lr`
- 增加训练步数 `--steps_per_epoch`
- 检查环境奖励设计

### 3. 性能问题
- 启用 `--parallel_collect` 使用多进程
- 调整 `--num_workers_collect` 进程数
- 使用GPU训练

## 示例命令

### 快速测试
```bash
python train_rl_sb3_simple.py --epochs 10 --num_agents 16 --steps_per_epoch 1024
```

### 生产训练
```bash
python train_rl_sb3_simple.py --epochs 100 --num_agents 64 --steps_per_epoch 4096 --parallel_collect --num_workers_collect 8
```

### CPU训练
```bash
python train_rl_sb3_simple.py --force_cpu --epochs 50 --mini_batch_size 32
```





