import os
import torch
import numpy as np
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from train_args_sb3 import get_sb3_args
from MAPFenv import MAPFEnv


class MAPFGymEnv(gym.Env):
    """简化的MAPF Gym环境"""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # 创建MAPF环境
        self.env = MAPFEnv(**kwargs)
        
        # 观测空间
        obs_shape = self.env.get_obs().shape
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=obs_shape, 
            dtype=np.float32
        )
        
        # 动作空间：简化为选择一个全局策略
        # 这里我们将问题简化为选择128个预定义的策略之一
        self.action_space = spaces.Discrete(128)
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        obs, info = self.env.reset()
        return obs.astype(np.float32), info
    
    def step(self, action):
        # 将离散动作转换为2D动作地图
        # 这里使用简单的启发式方法
        height, width = self.env.height, self.env.width
        
        # 基于动作ID生成动作地图
        np.random.seed(action)  # 使用动作作为随机种子
        action_map = np.random.randint(0, 5, (height, width))
        
        obs, reward, done, truncated, info = self.env.step(action_map)
        
        # 确保reward是标量
        if isinstance(reward, np.ndarray):
            reward = reward.sum()
        
        return obs.astype(np.float32), float(reward), done, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """简单的CNN特征提取器"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算展平后的大小
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.zeros(1, *observation_space.shape)
            ).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class ProgressCallback(BaseCallback):
    """训练进度回调"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # 检查episode结束
        for idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                self.episode_count += 1
                if 'infos' in self.locals and idx < len(self.locals['infos']):
                    info = self.locals['infos'][idx]
                    if 'episode' in info:
                        episode_reward = info['episode']['r']
                        episode_length = info['episode']['l']
                        if self.verbose > 0:
                            print(f"Episode {self.episode_count}: "
                                f"Reward={episode_reward:.2f}, Length={episode_length}")
        return True


def make_env(env_kwargs):
    """环境工厂函数"""
    def _init():
        env = MAPFGymEnv(**env_kwargs)
        env = Monitor(env)
        return env
    return _init


def train_sb3_simple(args):
    """使用Stable Baselines3进行简化训练"""
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建日志目录
    args.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"SB3_Simple_RL_{args.current_time}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 环境配置
    env_kwargs = {
        'height': 32,
        'width': 32,
        'num_agents': args.num_agents,
        'obstacle_density': 0.2,
        'max_steps': args.max_episode_steps,
        'feature_dim': args.feature_dim,
        'feature_type': args.feature_type
    }
    
    # 创建环境
    env = DummyVecEnv([make_env(env_kwargs)])
    eval_env = DummyVecEnv([make_env(env_kwargs)])
    
    # 策略配置
    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])],
        activation_fn=nn.ReLU,
    )
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    
    # 创建PPO模型
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.pi_lr,
        n_steps=args.steps_per_epoch,
        batch_size=args.mini_batch_size,
        n_epochs=args.train_pi_iters,
        gamma=args.gamma,
        gae_lambda=args.lam,
        clip_range=args.clip_ratio,
        ent_coef=args.entropy_coef,
        vf_coef=1.0,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=log_dir
    )
    
    # 设置回调函数
    callbacks = []
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_interval * args.steps_per_epoch,
        save_path=log_dir,
        name_prefix="rl_model"
    )
    callbacks.append(checkpoint_callback)
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=args.eval_interval * args.steps_per_epoch,
        n_eval_episodes=args.num_eval_episodes,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # 进度回调
    progress_callback = ProgressCallback(verbose=1)
    callbacks.append(progress_callback)
    
    # 开始训练
    total_timesteps = args.epochs * args.steps_per_epoch
    print(f"开始训练，总步数: {total_timesteps}")
    print(f"使用设备: {device}")
    print(f"日志目录: {log_dir}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = os.path.join(log_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"训练完成！最终模型保存至: {final_model_path}")
    
    # 清理环境
    env.close()
    eval_env.close()
    
    return model, log_dir


def test_model(model_path, env_kwargs, num_episodes=5):
    """测试训练好的模型"""
    
    # 创建环境
    env = DummyVecEnv([make_env(env_kwargs)])
    
    # 加载模型
    model = PPO.load(model_path)
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            step_count += 1
            
            if done[0]:
                break
            
            if step_count > 500:  # 防止死循环
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward={episode_reward:.2f}, Steps={step_count}")
    
    print(f"\n平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    env.close()


if __name__ == "__main__":
    args = get_sb3_args()
    
    print("=== Stable Baselines3 简化版MAPF训练 ===")
    print(f"环境: MAPF ({args.num_agents} agents)")
    print(f"算法: PPO")
    print(f"设备: {'CUDA' if torch.cuda.is_available() and not args.force_cpu else 'CPU'}")
    print("动作空间: 简化为离散选择")
    print("========================================")
    
    try:
        model, log_dir = train_sb3_simple(args)
        
        # 测试训练好的模型
        print("\n开始测试训练好的模型...")
        model_path = os.path.join(log_dir, "final_model.zip")
        env_kwargs = {
            'height': 32, 'width': 32, 'num_agents': args.num_agents,
            'obstacle_density': 0.2, 'max_steps': args.max_episode_steps,
            'feature_dim': args.feature_dim, 'feature_type': args.feature_type
        }
        test_model(model_path, env_kwargs)
            
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()





