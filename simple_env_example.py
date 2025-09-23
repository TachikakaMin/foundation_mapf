"""
简化MAPF环境使用示例
展示如何使用重写后的MAPFenv进行训练和测试
"""

import torch
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MAPFenv import MAPFEnv
from tools.path_formation import sample_action




def filter_wall_collisions(env, action_probabilities):
    """
    过滤掉会撞墙的动作概率
    
    参数:
        env: MAPFEnv环境对象
        action_probabilities: 每个智能体的动作概率 [num_agents, 5]
    
    返回:
        filtered_probabilities: 过滤后的动作概率 [num_agents, 5]
    """
    device = action_probabilities.device
    filtered_probs = action_probabilities.clone()
    
    # 定义5个动作的移动向量: [停留, 上, 下, 左, 右]
    action_deltas = torch.tensor([
        [0, 0],   # 0: 停留
        [0, 1],  # 1: 上 (减少x坐标)
        [0, -1],   # 2: 下 (增加x坐标)
        [-1, 0],  # 3: 左 (减少x坐标)
        [1, 0]    # 4: 右 (增加x坐标)
    ], device=device)
    
    for i, current_pos in enumerate(env.agent_positions):
        # 确保current_pos与action_deltas在同一设备上
        current_pos = current_pos.to(device)
        for action_idx in range(5):  # 检查所有5个动作
            # 计算执行该动作后的新位置
            new_pos = current_pos + action_deltas[action_idx]
            
            # 检查是否超出地图边界
            if (new_pos[0] < 0 or new_pos[0] >= env.height or 
                new_pos[1] < 0 or new_pos[1] >= env.width):
                filtered_probs[i, action_idx] = 0.0
                continue
            
            # 检查是否撞到障碍物
            if hasattr(env, 'map_data') and env.map_data is not None:
                if env.map_data[new_pos[0], new_pos[1]] == 1:  # 1表示障碍物
                    filtered_probs[i, action_idx] = 0.0
                    continue
            
            # 如果环境有障碍物地图属性
            if hasattr(env, 'obstacle_map') and env.obstacle_map is not None:
                if env.obstacle_map[new_pos[0], new_pos[1]] == 1:
                    filtered_probs[i, action_idx] = 0.0
                    continue
    
    # 确保每个agent至少有一个有效动作（停留动作总是有效的，除非当前位置就是障碍物）
    for i in range(filtered_probs.shape[0]):
        if filtered_probs[i].sum() == 0:
            # 如果所有动作都被过滤掉了，至少保留停留动作
            filtered_probs[i, 0] = 1e-6  # 给停留动作一个很小的概率
    
    # 重新归一化概率
    row_sums = filtered_probs.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-8)  # 避免除零
    filtered_probs = filtered_probs / row_sums
    
    return filtered_probs


def refine_actions_with_custom_pibt(env, action_probabilities, seed=0):
    """
    使用自定义单步PIBT算法来改进模型输出的动作概率
    
    参数:
        env: MAPFEnv环境对象
        action_probabilities: 每个智能体的动作概率 [num_agents, 5]
        seed: 随机种子
    
    返回:
        improved_actions: PIBT优化后的动作索引 [num_agents]
    """
    # 尝试导入自定义的PIBT模块
    import sys
    import os
    pibt_path = os.path.join(os.path.dirname(__file__), "pibt2")
    if pibt_path not in sys.path:
        sys.path.insert(0, pibt_path)
    
    from pibt_wrapper import pibt_solve_single_step
    
    # 使用自定义PIBT求解
    actions = pibt_solve_single_step(env, action_probabilities, seed=seed)
    
    return actions
        


def convert_actions_to_action_map(env, actions):
    """
    将动作索引数组转换为action_map格式
    
    参数:
        env: MAPFEnv环境对象
        actions: 每个智能体的动作索引 [num_agents]
    
    返回:
        action_map: 地图格式的动作 [height, width]
    """
    device = env.agent_positions.device
    action_map = torch.zeros(env.height, env.width, dtype=torch.long, device=device)
    
    # 将每个智能体的动作设置到其当前位置上
    for i, action in enumerate(actions):
        pos = env.agent_positions[i]
        action_map[pos[0], pos[1]] = action
    
    return action_map


def pibt_action_refinement(env, action_probabilities, seed=0, sampling=True):
    """
    便捷函数：使用PIBT改进动作并转换为action_map格式
    
    参数:
        env: MAPFEnv环境对象
        action_probabilities: 每个智能体的动作概率 [num_agents, 5]
        seed: 随机种子
        sampling: 是否使用采样 (仅对原始PIBT有效)
    
    返回:
        action_map: 可直接用于env.step()的action_map [height, width]
    """
    # 使用PIBT改进动作
    improved_actions = refine_actions_with_custom_pibt(env, action_probabilities, seed=seed)
    # 转换为action_map格式
    action_map = convert_actions_to_action_map(env, improved_actions)
    
    return action_map


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


def example_with_pretrained_model(model_path, feature_dim=4, first_layer_channels=64, bilinear=False, test_pibt=False, save_video=False, output_dir="videos"):
    """示例: 使用预训练模型"""
    from models.unet import UNet
    max_episode_steps = 20
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
        
        if args.test_pibt:
            # 从logits中获取每个智能体位置的动作概率
            agent_positions = env.agent_positions  # [num_agents, 2]
            
            # 提取每个智能体位置的logits作为动作概率
            action_probabilities = torch.zeros(env.num_agents, 5, device=device)
            for i, pos in enumerate(agent_positions):
                action_probabilities[i] = logits[0, :, pos[0], pos[1]]
            
            # 应用softmax归一化
            action_probabilities = torch.softmax(action_probabilities, dim=-1)
            
            # ⭐ 过滤掉会撞墙的动作
            filtered_action_probabilities = filter_wall_collisions(env, action_probabilities)
            
            # 计算过滤统计信息
            original_valid = (action_probabilities > 1e-6).sum().item()
            filtered_valid = (filtered_action_probabilities > 1e-6).sum().item()
            
            # 使用过滤后的动作概率进行PIBT改进并转换为action_map格式
            action_map = pibt_action_refinement(env, filtered_action_probabilities, seed=step, sampling=True)
            
            print(f"步骤 {step + 1:2d}: 过滤前有效动作={original_valid}, 过滤后有效动作={filtered_valid}, action_map形状={action_map.shape}")
        
        else:
            # 从logits中采样动作
            feature = env.get_feature()
            # 确保feature在与logits相同的设备上
            if isinstance(feature, torch.Tensor):
                feature = feature.to(device)
            elif isinstance(feature, (list, tuple)):
                feature = [f.to(device) if isinstance(f, torch.Tensor) else f for f in feature]
            
            action_map = sample_action(
                logits,
                env.agent_positions, 
                env.temperature,
                feature,
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
    parser.add_argument(
        "--test_pibt", 
        action="store_true", 
        default=False,
        help="是否运行PIBT动作改进测试"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    
    # 检查模型文件是否存在
    if os.path.exists(args.model_path):
        print("=== 预训练模型测试 ===")
        example_with_pretrained_model(
            model_path=args.model_path,
            feature_dim=args.feature_dim,
            first_layer_channels=args.first_layer_channels,
            bilinear=args.bilinear,
            test_pibt=args.test_pibt,
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