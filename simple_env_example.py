"""
简化MAPF环境使用示例
展示如何使用重写后的MAPFenv进行训练和测试
"""

import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MAPFenv import MAPFEnv
from tools.path_formation import sample_action


def example_usage():
    """示例: 如何使用简化的MAPF环境"""
    
    # 创建环境
    env = MAPFEnv(
        height=32,          # 地图高度
        width=32,           # 地图宽度  
        num_agents=8,       # 代理数量
        obstacle_density=0.2, # 障碍物密度
        max_steps=100,      # 最大步数
        feature_dim=6,      # 特征维度
        feature_type="gradient"  # 特征类型
    )
    
    print("环境创建完成!")
    
    # 重置环境
    obs, info = env.reset()
    print(f"观测形状: {obs.shape}")  # (4, 32, 32)
    print(f"初始成功率: {info['success_rate']:.2%}")
    
    # 模拟一个简单的训练循环
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 20:  # 最多20步
        step += 1
        
        # 模拟神经网络输出 (这里用随机值代替)
        # 实际使用时，这应该是你的神经网络的输出
        network_output = torch.randn(5, env.height, env.width)
        
        # 从logits中采样动作
        from tools.path_formation import sample_action
        action_map = sample_action(
            network_output.unsqueeze(0),  # 添加batch维度
            env.agent_positions, 
            env.temperature,
            env.get_feature(),  # 需要feature来创建mask
            action_choice="sample"
        )
        
        # 执行步骤
        next_obs, reward, done, truncated, info = env.step(action_map)
        
        total_reward += reward
        
        print(f"步骤 {step}: 奖励={reward:.2f}, 成功率={info['success_rate']:.2%}")
        
        if done:
            print("🎉 所有代理都到达目标!")
            break
        
        if truncated:
            print("⏰ 达到最大步数限制")
            break
    
    print(f"\n总奖励: {total_reward:.2f}")
    print(f"最终成功率: {info['success_rate']:.2%}")
    
    # 渲染最终状态
    env.render()


def save_frames_as_video(frames, output_path, fps=5, format='mp4'):
    """将帧序列保存为视频"""
    if not frames:
        print("没有帧可保存")
        return
    
    print(f"保存 {len(frames)} 帧到视频: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 设置动画
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    
    # 显示第一帧
    im = ax.imshow(frames[0])
    
    def animate(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                 interval=1000//fps, blit=True, repeat=True)
    
    # 尝试保存为MP4
    saved = False
    if format.lower() == 'mp4':
        try:
            writer = animation.FFMpegWriter(
                fps=fps,
                metadata=dict(artist='MAPF-RL'),
                codec='h264',
                bitrate=1500,
                extra_args=['-pix_fmt', 'yuv420p']
            )
            anim.save(output_path, writer=writer)
            print(f"✅ MP4保存成功: {output_path}")
            saved = True
        except Exception as e:
            print(f"⚠️ MP4保存失败: {e}")
            # 如果h264失败，尝试mpeg4
            try:
                writer = animation.FFMpegWriter(
                    fps=fps,
                    metadata=dict(artist='MAPF-RL'),
                    codec='mpeg4',
                    bitrate=1200
                )
                anim.save(output_path, writer=writer)
                print(f"✅ MP4保存成功 (mpeg4): {output_path}")
                saved = True
            except Exception as e2:
                print(f"⚠️ mpeg4也失败: {e2}")
    
    # 如果MP4失败，保存为GIF
    if not saved:
        gif_path = output_path.replace('.mp4', '.gif').replace('.avi', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=fps)
            print(f"✅ 保存为GIF: {gif_path}")
            saved = True
        except Exception as e:
            print(f"❌ GIF保存也失败: {e}")
    
    if not saved:
        print("❌ 所有视频格式保存都失败了")
    
    plt.close(fig)


def example_with_pretrained_model(model_path, feature_dim=4, first_layer_channels=64, bilinear=False, save_video=False, output_dir="videos"):
    """示例: 使用预训练模型"""
    from models.unet import UNet
    max_episode_steps = 1000
    print(f"加载模型: {model_path}")
    
    # 创建环境 - 使用较小的尺寸以便观察
    env = MAPFEnv(
        height=32, 
        width=32, 
        num_agents=64, 
        obstacle_density=0.15,
        max_steps=max_episode_steps,
        feature_dim=feature_dim
    )
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = UNet(
        n_channels=feature_dim, 
        n_classes=5, 
        first_layer_channels=first_layer_channels,
        bilinear=bilinear
    ).to(device)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 重置环境
    obs, info = env.reset()
    print(f"初始观测形状: {obs.shape}")
    print(f"初始成功率: {info['success_rate']:.2%}")
    
    # 收集帧用于视频
    frames = []
    if save_video:
        # 添加初始帧 - 使用较低分辨率减少文件大小
        initial_frame = env.render_frame(dpi=60, figsize=(6, 6))
        frames.append(initial_frame)
        print("开始收集帧用于视频生成...")
    
    total_reward = 0
    success_history = []
    
    for step in range(max_episode_steps):  # 运行50步
        # 将观测转换为模型输入
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)  # [1, feature_dim, H, W]
        
        # 获取模型输出
        with torch.no_grad():
            logits, _ = model(obs_tensor)  # [1, 5, H, W]
        
        # 从logits中采样动作
        action_map = sample_action(
            logits,
            env.agent_positions, 
            env.temperature,
            env.get_feature(),
            action_choice="sample"
        )
        
        # 环境执行步骤
        next_obs, reward, done, truncated, info = env.step(action_map)
        
        total_reward += reward
        success_rate = info['success_rate']
        success_history.append(success_rate)
        
        print(f"步骤 {step + 1:2d}: 奖励={reward:8.4f}, 成功={info['success_count']}/{env.num_agents} ({success_rate:5.1%})")
        
        # 收集帧 - 使用较低分辨率减少文件大小
        if save_video:
            frame = env.render_frame(dpi=60, figsize=(6, 6))
            frames.append(frame)
        
        obs = next_obs
        
        if done:
            print("🎉 所有代理都到达目标!")
            break
        
        if truncated:
            print("⏰ 达到最大步数限制")
            break
    
    print(f"\n=== 测试结果 ===")
    print(f"总奖励: {total_reward:.4f}")
    print(f"最终成功率: {info['success_rate']:.1%}")
    print(f"最高成功率: {max(success_history):.1%}")
    
    # 生成视频
    if save_video and frames:
        video_filename = f"mapf_test_{env.num_agents}agents.mp4"
        video_path = os.path.join(output_dir, video_filename)
        save_frames_as_video(frames, video_path, fps=5)
    
    # 渲染最终状态
    if not save_video:
        env.render()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MAPF环境测试")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="saved_model/epoch_4_bs64_bilinearF_flc_64_7Mdata.pth",
        help="预训练模型路径"
    )
    parser.add_argument(
        "--feature_dim", 
        type=int, 
        default=4,
        help="特征维度"
    )
    parser.add_argument(
        "--first_layer_channels", 
        type=int, 
        default=64,
        help="第一层通道数"
    )
    parser.add_argument(
        "--bilinear", 
        action="store_true", 
        default=False,
        help="是否使用双线性上采样"
    )
    parser.add_argument(
        "--test_random", 
        action="store_true", 
        default=False,
        help="是否运行随机动作测试"
    )
    parser.add_argument(
        "--save_video", 
        action="store_true", 
        default=False,
        help="是否生成视频"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="videos",
        help="视频输出目录"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.test_random:
        print("=== 随机动作测试 ===")
        example_usage()
        print()
    
    # 检查模型文件是否存在
    if os.path.exists(args.model_path):
        print("=== 预训练模型测试 ===")
        example_with_pretrained_model(
            model_path=args.model_path,
            feature_dim=args.feature_dim,
            first_layer_channels=args.first_layer_channels,
            bilinear=args.bilinear,
            save_video=args.save_video,
            output_dir=args.output_dir
        )
    else:
        print(f"❌ 模型文件不存在: {args.model_path}")
        print("可用的模型文件:")
        if os.path.exists("saved_model"):
            for f in os.listdir("saved_model"):
                if f.endswith(".pth"):
                    print(f"  - saved_model/{f}")
        else:
            print("  没有找到saved_model目录") 