import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from tqdm import tqdm
from train_args import get_rl_args
from models.unet import UNet
from models.CNN import CNN
from torch.utils.tensorboard import SummaryWriter
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import deque
import psutil
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import copy

# MAPF强化学习环境
from MAPFenv import MAPFEnv


class PPOBuffer:
    """PPO经验回放缓冲区（线程安全版本）"""
    
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device='cpu'):
        self.obs_buf = torch.zeros((size, *obs_dim), dtype=torch.float32).to(device)
        # 对于中心化策略，动作是整个地图的动作 [H, W]
        self.act_buf = torch.zeros((size, obs_dim[1], obs_dim[2]), dtype=torch.long).to(device)  # [size, H, W]
        self.adv_buf = torch.zeros(size, dtype=torch.float32).to(device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32).to(device)
        self.ret_buf = torch.zeros(size, dtype=torch.float32).to(device)
        self.val_buf = torch.zeros(size, dtype=torch.float32).to(device)
        self.logp_buf = torch.zeros((size, obs_dim[1], obs_dim[2]), dtype=torch.float32).to(device)
        # 对于中心化策略，mask也是整个地图的mask [H, W]
        self.mask_buf = torch.zeros((size, obs_dim[1], obs_dim[2]), dtype=torch.bool).to(device)  # [size, H, W]
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device
        
        # 线程安全锁
        self.lock = threading.Lock()
        
        # 路径跟踪（用于多线程环境）
        self.path_starts = []

    def store(self, obs, act, rew, val, logp, mask):
        """存储一步经验（线程安全）"""
        with self.lock:
            if self.ptr >= self.max_size:
                return False  # 缓冲区已满
            
            self.obs_buf[self.ptr] = obs.to(self.device)
            self.act_buf[self.ptr] = act.to(self.device)
            self.rew_buf[self.ptr] = rew
            self.val_buf[self.ptr] = val.to(self.device)
            self.logp_buf[self.ptr] = logp.to(self.device)
            self.mask_buf[self.ptr] = mask.to(self.device)
            self.ptr += 1
            return True

    def store_batch(self, data_list):
        """批量存储经验（线程安全）"""
        with self.lock:
            stored_count = 0
            start_ptr = self.ptr
            for obs, act, rew, val, logp, mask in data_list:
                if self.ptr >= self.max_size:
                    break
                
                self.obs_buf[self.ptr] = obs.to(self.device)
                self.act_buf[self.ptr] = act.to(self.device)
                self.rew_buf[self.ptr] = rew
                self.val_buf[self.ptr] = val.to(self.device)
                self.logp_buf[self.ptr] = logp.to(self.device)
                self.mask_buf[self.ptr] = mask.to(self.device)
                self.ptr += 1
                stored_count += 1
            return stored_count, start_ptr

    def mark_path_start(self):
        """标记新轨迹的开始"""
        with self.lock:
            self.path_starts.append(self.ptr)

    def finish_path(self, last_val=0):
        """完成一个轨迹，计算优势和回报"""
        with self.lock:
            path_slice = slice(self.path_start_idx, self.ptr)
            if path_slice.start >= path_slice.stop:
                return  # 空路径
            
            rews = torch.cat([self.rew_buf[path_slice], torch.tensor([last_val]).to(self.device)])
            vals = torch.cat([self.val_buf[path_slice], torch.tensor([last_val]).to(self.device)])
            
            # GAE-Lambda 优势估计
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
            
            # 计算回报 (回报到未来)
            self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
            
            self.path_start_idx = self.ptr

    def finish_paths_batch(self, path_info_list):
        """批量完成多个轨迹"""
        with self.lock:
            for start_idx, end_idx, last_val in path_info_list:
                if start_idx >= end_idx:
                    continue
                
                path_slice = slice(start_idx, end_idx)
                rews = torch.cat([self.rew_buf[path_slice], torch.tensor([last_val]).to(self.device)])
                vals = torch.cat([self.val_buf[path_slice], torch.tensor([last_val]).to(self.device)])
                
                # GAE-Lambda 优势估计
                deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
                self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
                
                # 计算回报
                self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

    def get(self, target_device=None):
        """获取缓冲区中的所有数据"""
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        # 标准化优势 - 对于中心化策略，我们仍然使用标量优势
        adv_mean = self.adv_buf.mean()
        adv_std = self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                   adv=self.adv_buf, logp=self.logp_buf, mask=self.mask_buf)
        
        # 如果指定了目标设备，则将数据移动到该设备
        if target_device is not None and target_device != self.device:
            data = {k: v.to(target_device) for k, v in data.items()}
            
        return data

    def _discount_cumsum(self, x, discount):
        """计算折扣累积和"""
        result = torch.zeros_like(x)
        result[-1] = x[-1]
        for i in reversed(range(len(x) - 1)):
            result[i] = x[i] + discount * result[i + 1]
        return result


class PPOActorCritic(nn.Module):
    """PPO Actor-Critic网络"""
    
    def __init__(self, obs_dim, act_dim, model_type='unet', **kwargs):
        super().__init__()
        
        # 策略网络(Actor) - 重用原有的模型架构
        if model_type == "unet":
            self.policy = UNet(n_channels=obs_dim[0], n_classes=act_dim, **kwargs)
        elif model_type == "cnn":
            self.policy = CNN(n_channels=obs_dim[0], n_classes=act_dim)
        
        # 价值网络(Critic) - 输出单一价值
        if model_type == "unet":
            self.value_net = UNet(n_channels=obs_dim[0], n_classes=1, **kwargs)
        elif model_type == "cnn":
            self.value_net = CNN(n_channels=obs_dim[0], n_classes=1)

    def step(self, obs, mask=None):
        """给定观测，输出动作、价值和对数概率"""
        with torch.no_grad():
            logits, _ = self.policy(obs)  # [batch, act_dim, H, W]
            values = self.value_net(obs)[0].squeeze(1)  # [batch, H, W]
            
            # 应用mask（只在有代理的位置采样动作）
            if mask is not None:
                # 使用更稳定的mask方法
                mask_expanded = mask.unsqueeze(1).expand_as(logits)
                logits = torch.where(mask_expanded, logits, torch.full_like(logits, -10.0))  # 使用-10而不是-inf
                values = values * mask.float()
            
            # 限制logits范围，防止数值不稳定
            logits = torch.clamp(logits, -10.0, 10.0)
            
            # 从logits采样动作
            dist = torch.distributions.Categorical(logits=logits.permute(0, 2, 3, 1))  # [batch, H, W, act_dim]
            actions = dist.sample()  # [batch, H, W]
            logp_a = dist.log_prob(actions)  # [batch, H, W]
            
            # 只返回有效位置的值
            if mask is not None:
                logp_a = logp_a * mask.float()
                
        return actions, values, logp_a

    def act(self, obs, mask=None):
        """仅用于推理的动作选择"""
        return self.step(obs, mask)[0]

    def evaluate(self, obs, act, mask=None):
        """评估动作的价值和对数概率"""
        logits, _ = self.policy(obs)  # [batch, act_dim, H, W]  
        values = self.value_net(obs)[0].squeeze(1)  # [batch, H, W]
        
        if mask is not None:
            # 使用更稳定的mask方法
            mask_expanded = mask.unsqueeze(1).expand_as(logits)
            logits = torch.where(mask_expanded, logits, torch.full_like(logits, -10.0))
            values = values * mask.float()
        
        # 限制logits范围
        logits = torch.clamp(logits, -10.0, 10.0)
        
        dist = torch.distributions.Categorical(logits=logits.permute(0, 2, 3, 1))
        logp_a = dist.log_prob(act)
        entropy = dist.entropy()
        
        if mask is not None:
            logp_a = logp_a * mask.float()
            entropy = entropy * mask.float()
            
        return values, logp_a, entropy


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, actor_critic, args, device):
        self.ac = actor_critic
        self.args = args
        self.device = device
        
        # 优化器
        self.pi_optimizer = torch.optim.AdamW(
            self.ac.policy.parameters(), 
            lr=args.pi_lr, 
            weight_decay=args.weight_decay
        )
        self.vf_optimizer = torch.optim.AdamW(
            self.ac.value_net.parameters(), 
            lr=args.vf_lr, 
            weight_decay=args.weight_decay
        )
        
        # PPO参数
        self.clip_ratio = args.clip_ratio
        self.train_pi_iters = args.train_pi_iters
        self.train_v_iters = args.train_v_iters
        self.target_kl = args.target_kl
        self.mini_batch_size = args.mini_batch_size

    def _get_mini_batches(self, data):
        """将数据分割成mini-batch"""
        batch_size = data['obs'].shape[0]
        indices = torch.randperm(batch_size)
        
        for start_idx in range(0, batch_size, self.mini_batch_size):
            end_idx = min(start_idx + self.mini_batch_size, batch_size)
            batch_indices = indices[start_idx:end_idx]
            
            mini_batch = {}
            for key, value in data.items():
                mini_batch[key] = value[batch_indices]
            
            yield mini_batch

    def update(self, buffer):
        """PPO策略更新（使用mini-batch）"""
        data = buffer.get(target_device=self.device)
        
        # 用第一个mini-batch计算初始损失（避免800样本的GPU内存消耗）
        first_mini_batch = next(self._get_mini_batches(data))
        pi_l_old, pi_info_old = self._compute_loss_pi(first_mini_batch)
        v_l_old = self._compute_loss_v(first_mini_batch)
        
        # 累积统计信息
        total_pi_loss, total_v_loss = 0.0, 0.0
        total_kl, total_entropy, total_clip_frac = 0.0, 0.0, 0.0
        update_count = 0
        
        # 策略网络更新
        for i in range(self.train_pi_iters):
            batch_losses, batch_kl, batch_entropy, batch_clip_frac = [], 0.0, 0.0, 0.0
            batch_count = 0
            
            for mini_batch in self._get_mini_batches(data):
                self.pi_optimizer.zero_grad()
                loss_pi, pi_info = self._compute_loss_pi(mini_batch)
                
                # 检查损失是否为NaN
                if torch.isnan(loss_pi):
                    print(f"⚠️  警告: 策略损失为NaN，跳过这个mini-batch")
                    continue
                
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.policy.parameters(), 0.1)
                self.pi_optimizer.step()
                
                batch_losses.append(loss_pi.item())
                batch_kl += pi_info.get('kl', 0)
                batch_entropy += pi_info.get('entropy', 0)
                batch_clip_frac += pi_info.get('clip_frac', 0)
                batch_count += 1
            
            if batch_count > 0:
                avg_kl = batch_kl / batch_count
                if avg_kl > 1.5 * self.target_kl:
                    if hasattr(self, 'args') and self.args.local_rank == 0:
                        print(f'Early stopping at step {i} due to reaching max kl.')
                    break
                
                total_kl += avg_kl
                total_entropy += batch_entropy / batch_count
                total_clip_frac += batch_clip_frac / batch_count
                update_count += 1
        
        # 价值网络更新
        for i in range(self.train_v_iters):
            batch_losses = []
            
            for mini_batch in self._get_mini_batches(data):
                self.vf_optimizer.zero_grad()
                loss_v = self._compute_loss_v(mini_batch)
                
                # 检查损失是否为NaN
                if torch.isnan(loss_v):
                    print(f"⚠️  警告: 价值损失为NaN，跳过这个mini-batch")
                    continue
                    
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.value_net.parameters(), 0.1)
                self.vf_optimizer.step()
                
                batch_losses.append(loss_v.item())
            
            if batch_losses:
                total_v_loss += sum(batch_losses) / len(batch_losses)
        
        # 计算平均统计信息
        avg_stats = {
            'kl': total_kl / max(update_count, 1),
            'entropy': total_entropy / max(update_count, 1), 
            'clip_frac': total_clip_frac / max(update_count, 1),
            'pi_loss': pi_l_old.item(),
            'v_loss': v_l_old.item()
        }
        
        return avg_stats

    def _compute_loss_pi(self, data):
        """计算策略损失（中心化策略）"""
        obs, act, adv, logp_old, mask = data['obs'], data['act'], data['adv'], data['logp'], data['mask']
        
        # 策略损失 - act现在是[batch, H, W]格式，mask也是[batch, H, W]格式
        _, logp, entropy = self.ac.evaluate(obs, act, mask)
        
        # 使用存储的agent mask
        valid_positions = mask.float()
        
        # 计算每个位置的损失
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv.unsqueeze(-1).unsqueeze(-1)  # 将标量adv扩展到[H, W]
        loss_pi = -(torch.min(ratio * adv.unsqueeze(-1).unsqueeze(-1), clip_adv) * valid_positions).sum() / valid_positions.sum()
        
        # 有用的额外信息
        approx_kl = ((logp_old - logp) * valid_positions).sum() / valid_positions.sum()
        entropy_loss = (entropy * valid_positions).sum() / valid_positions.sum()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clip_frac = (clipped.float() * valid_positions).sum() / valid_positions.sum()
        
        # 加入熵奖励
        loss_pi -= self.args.entropy_coef * entropy_loss
        
        pi_info = dict(kl=approx_kl.item(), entropy=entropy_loss.item(), clip_frac=clip_frac.item())
        
        return loss_pi, pi_info

    def _compute_loss_v(self, data):
        """计算价值损失（中心化策略）"""
        obs, ret, mask = data['obs'], data['ret'], data['mask']
        values = self.ac.value_net(obs)[0].squeeze(1)  # [batch, H, W]
        
        # 使用存储的agent mask - mask现在是[batch, H, W]格式
        valid_positions = mask.float()
        
        # 将标量return扩展到agent位置
        ret_expanded = ret.unsqueeze(-1).unsqueeze(-1).expand_as(values)
        
        # 计算价值损失
        loss_v = ((values - ret_expanded) ** 2 * valid_positions).sum() / valid_positions.sum()
        
        return loss_v


def worker_collect_experience(worker_id, env_config, actor_critic_state, steps_to_collect, device, results_queue, progress_queue):
    """工作线程函数：收集经验"""
    # 在工作线程中强制使用CPU，避免多线程GPU冲突
    worker_device = torch.device('cpu')
    
    # 创建独立的环境实例（使用预计算的地图和距离图）
    env = MAPFEnv(**env_config)
    
    # 设置不同的随机种子以确保多样性
    seed = worker_id * 1000 + np.random.randint(0, 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 为当前线程复制模型（保持在CPU以避免CUDA上下文问题）
    actor_critic_worker = copy.deepcopy(actor_critic_state)
    actor_critic_worker.to(worker_device)
    actor_critic_worker.eval()
    
    # 收集的经验数据
    experiences = []
    path_infos = []
    episode_returns = []
    
    obs, info = env.reset()
    ep_ret, ep_len = 0, 0
    path_start = 0
    
    for t in range(steps_to_collect):
        # 转换观测格式 - 在工作线程中使用CPU
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=worker_device).unsqueeze(0)
        mask_tensor = torch.as_tensor(info.get('agent_mask', np.ones_like(obs[0], dtype=bool)), 
                                    dtype=torch.bool, device=worker_device).unsqueeze(0)
        
        # 获取动作、价值和对数概率
        with torch.no_grad():
            logits, _ = actor_critic_worker.policy(obs_tensor)
            val = actor_critic_worker.value_net(obs_tensor)[0].squeeze(1)
        
        # 使用sample_action采样动作 - 确保所有张量在同一设备
        from tools.path_formation import sample_action
        action_map = sample_action(
            logits.cpu(),  # 将logits移到CPU以匹配环境
            env.agent_positions, 
            env.temperature,
            env.get_feature(),  # 获取当前特征
            action_choice="sample"
        )
        
        # 计算动作的对数概率（用于存储）
        with torch.no_grad():
            # 应用mask
            if mask_tensor is not None:
                mask_expanded = mask_tensor.unsqueeze(1).expand_as(logits)
                logits_masked = torch.where(mask_expanded, logits, torch.full_like(logits, -10.0))
            else:
                logits_masked = logits
            
            logits_masked = torch.clamp(logits_masked, -10.0, 10.0)
            dist = torch.distributions.Categorical(logits=logits_masked.permute(0, 2, 3, 1))
            logp = dist.log_prob(action_map.to(worker_device).unsqueeze(0))  # 确保action_map在CPU
            
            if mask_tensor is not None:
                logp = logp * mask_tensor.float()

            logp_map = logp[0]
        
        # 执行动作
        next_obs, reward, done, truncated, next_info = env.step(action_map)
        
        # 中心化策略：直接存储全局经验
        # 对于中心化策略，我们存储整个状态-动作对，而不是按agent聚合
        
        # 全局价值（有agent位置的平均值）
        agent_positions = torch.where(mask_tensor[0])
        if len(agent_positions[0]) > 0:
            # 只计算有agent位置的价值和对数概率
            active_values = val[0][agent_positions]
            active_logprobs = logp_map[agent_positions]

            global_value = active_values.mean()
            
            # 存储中心化经验
            experiences.append((
                obs_tensor[0],          # 完整环境观测 [channels, H, W] - 已在CPU
                action_map,             # 完整动作地图 [H, W] - 已在CPU 
                reward if isinstance(reward, (int, float)) else reward.sum(),  # 全局奖励
                global_value,           # 全局价值 - 已在CPU
                logp_map,               # 对数概率地图 [H, W] - 已在CPU
                mask_tensor[0]          # agent位置的mask [H, W] - 已在CPU
            ))
        
        # 更新状态
        obs, info = next_obs, next_info
        ep_ret += reward if isinstance(reward, (int, float)) else reward.sum()
        ep_len += 1
        
        # 处理回合结束
        if done or truncated or t == steps_to_collect - 1:
            if not (done or truncated):
                # 估计最终价值
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=worker_device).unsqueeze(0)
                mask_tensor = torch.as_tensor(info.get('agent_mask', np.ones_like(obs[0], dtype=bool)), 
                                            dtype=torch.bool, device=worker_device).unsqueeze(0)
                with torch.no_grad():
                    last_val = actor_critic_worker.value_net(obs_tensor)[0].squeeze(1)
                last_val = last_val[0][mask_tensor[0]].mean().item()
            else:
                last_val = 0
            
            # 记录路径信息
            path_infos.append((path_start, len(experiences), last_val))
            episode_returns.append(ep_ret)
            
            if done or truncated:
                obs, info = env.reset()
            
            ep_ret, ep_len = 0, 0
            path_start = len(experiences)
        
        # 定期报告进度
        if t % 10 == 0:  # 每10步报告一次进度
            progress_queue.put((worker_id, t, len(episode_returns)))
    
    # 发送最终进度
    progress_queue.put((worker_id, steps_to_collect, len(episode_returns)))
    
    # 返回结果
    results_queue.put((worker_id, experiences, path_infos, episode_returns))


def collect_experience_parallel(env_template, actor_critic, buffer, steps_per_epoch, device, args):
    """并行收集经验"""
    num_workers = args.num_workers_collect
    steps_per_worker = steps_per_epoch // num_workers
    
    # 预先计算共享的地图和距离图（避免重复计算）
    if args.local_rank == 0:
        print("预计算距离图以供多线程共享...")
    
    shared_map_data = env_template.map_data.copy()
    shared_distance_map = env_template.distance_map  # 复用已有的距离图
    
    # 环境配置（包含预计算的地图和距离图）
    env_config = {
        'height': env_template.height,
        'width': env_template.width,
        'num_agents': env_template.num_agents,
        'obstacle_density': env_template.obstacle_density,
        'max_steps': env_template.max_steps,
        'feature_dim': env_template.feature_dim,
        'feature_type': env_template.feature_type,
        'map_data': shared_map_data,
        'distance_map': shared_distance_map
    }
    
    # 创建队列和线程
    results_queue = queue.Queue()
    progress_queue = queue.Queue()
    
    # 为了避免多线程GPU冲突，为工作线程准备一个CPU副本
    if isinstance(actor_critic, DDP):
        model_source = actor_critic.module
    else:
        model_source = actor_critic
    actor_critic_cpu = copy.deepcopy(model_source).to('cpu')
    
    # 创建进度条
    total_steps = steps_per_epoch
    step_pbar = tqdm(range(total_steps), desc="并行收集经验", leave=False, disable=args.local_rank != 0)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交工作任务
        futures = []
        for worker_id in range(num_workers):
            future = executor.submit(
                worker_collect_experience,
                worker_id, env_config, actor_critic_cpu, steps_per_worker,
                torch.device('cpu'), results_queue, progress_queue  # 强制使用CPU
            )
            futures.append(future)
        
        # 监控进度
        completed_steps = 0
        progress_counts = {i: 0 for i in range(num_workers)}
        finished_workers = set()
        
        # 实时监控进度并等待任务完成
        while len(finished_workers) < num_workers:
            # 处理进度更新
            while not progress_queue.empty():
                try:
                    worker_id, step, episodes = progress_queue.get_nowait()
                    if step > progress_counts[worker_id]:
                        step_increase = step - progress_counts[worker_id]
                        completed_steps += step_increase
                        progress_counts[worker_id] = step
                        step_pbar.update(step_increase)
                except queue.Empty:
                    break
            
            # 检查是否有完成的任务
            for i, future in enumerate(futures):
                if i not in finished_workers and future.done():
                    finished_workers.add(i)
                    future.result()  # 这会抛出工作线程中的异常（如果有的话）
            
            # 短暂休眠避免过于频繁的检查
            import time
            time.sleep(0.1)
    
    step_pbar.close()
    
    # 收集所有结果
    all_episode_returns = []
    total_stored = 0
    
    for _ in range(num_workers):
        result = results_queue.get()
        if len(result) == 5:  # 错误情况
            worker_id, experiences, path_infos, episode_returns, error = result
            print(f"工作线程 {worker_id} 出错: {error}")
            continue
        
        worker_id, experiences, path_infos, episode_returns = result
        
        if experiences is not None:
            # 批量存储经验
            stored_count, start_ptr = buffer.store_batch(experiences)
            total_stored += stored_count

            if stored_count > 0:
                # 将工作线程的局部索引映射到缓冲区的全局位置
                adjusted_paths = []
                for start_idx, end_idx, last_val in path_infos:
                    if start_idx >= stored_count:
                        continue
                    clipped_end = min(end_idx, stored_count)
                    if clipped_end <= start_idx:
                        continue
                    global_start = start_ptr + start_idx
                    global_end = start_ptr + clipped_end
                    adjusted_paths.append((global_start, global_end, last_val))

                if adjusted_paths:
                    buffer.finish_paths_batch(adjusted_paths)
            
            # 收集回合回报
            all_episode_returns.extend(episode_returns)
                
    
    if args.local_rank == 0:
        print(f"并行收集完成: 存储了 {total_stored} 个经验, 完成了 {len(all_episode_returns)} 个回合 (共享距离图，避免重复计算)")
    
    return all_episode_returns


def collect_experience(env, actor_critic, buffer, steps_per_epoch, device, args):
    """收集经验"""
    obs, info = env.reset()
    ep_ret, ep_len = 0, 0
    episode_returns = []
    
    step_pbar = tqdm(range(steps_per_epoch), desc="收集经验", leave=False, disable=args.local_rank != 0)
    
    for t in step_pbar:
        # 转换观测格式
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_tensor = torch.as_tensor(info.get('agent_mask', np.ones_like(obs[0], dtype=bool)), 
                                    dtype=torch.bool, device=device).unsqueeze(0)
        
        # 获取动作、价值和对数概率
        with torch.no_grad():
            logits, _ = actor_critic.policy(obs_tensor)
            val = actor_critic.value_net(obs_tensor)[0].squeeze(1)
        
        # 使用sample_action采样动作
        from tools.path_formation import sample_action
        action_map = sample_action(
            logits.cpu(),  # 将logits移到CPU以匹配环境
            env.agent_positions, 
            env.temperature,
            env.get_feature(),
            action_choice="sample"
        )
        
        # 计算动作的对数概率（用于存储）
        with torch.no_grad():
            # 应用mask
            if mask_tensor is not None:
                mask_expanded = mask_tensor.unsqueeze(1).expand_as(logits)
                logits_masked = torch.where(mask_expanded, logits, torch.full_like(logits, -10.0))
            else:
                logits_masked = logits
            
            logits_masked = torch.clamp(logits_masked, -10.0, 10.0)
            dist = torch.distributions.Categorical(logits=logits_masked.permute(0, 2, 3, 1))
            logp = dist.log_prob(action_map.to(device).unsqueeze(0))  # 确保action_map在正确设备
            
            if mask_tensor is not None:
                logp = logp * mask_tensor.float()

            logp_map = logp[0]
        
        # 执行动作
        next_obs, reward, done, truncated, next_info = env.step(action_map)
        
        # 中心化策略：直接存储全局经验
        # 对于中心化策略，我们存储整个状态-动作对，而不是按agent聚合
        
        # 全局价值（有agent位置的平均值）
        agent_positions = torch.where(mask_tensor[0])
        if len(agent_positions[0]) > 0:
            # 只计算有agent位置的价值和对数概率
            active_values = val[0][agent_positions]
            active_logprobs = logp_map[agent_positions]

            global_value = active_values.mean()
            
            # 存储中心化经验 - 确保所有数据在buffer的设备上
            buffer.store(
                obs_tensor[0].to(buffer.device),    # 完整环境观测 [channels, H, W]
                action_map.to(buffer.device),       # 完整动作地图 [H, W] 
                reward if isinstance(reward, (int, float)) else reward.sum(),  # 全局奖励
                global_value.to(buffer.device),     # 全局价值
                logp_map.to(buffer.device),         # 对数概率地图 [H, W]
                mask_tensor[0].to(buffer.device)    # agent位置的mask [H, W]
            )
        
        # 更新状态
        obs, info = next_obs, next_info
        ep_ret += reward if isinstance(reward, (int, float)) else reward.sum()
        ep_len += 1
        
        # 更新进度条信息
        if t % 100 == 0:  # 每100步更新一次显示，避免过于频繁
            step_pbar.set_postfix({
                'EpRet': f'{ep_ret:.2f}',
                'EpLen': f'{ep_len}',
                'Episodes': f'{len(episode_returns)}'
            })
        
        # 处理回合结束
        if done or truncated or t == steps_per_epoch - 1:
            if not (done or truncated):
                # 超时结束，需要估计最终价值
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_tensor = torch.as_tensor(info.get('agent_mask', np.ones_like(obs[0], dtype=bool)), 
                                            dtype=torch.bool, device=device).unsqueeze(0)
                with torch.no_grad():
                    last_val = actor_critic.value_net(obs_tensor)[0].squeeze(1)
                last_val = last_val[0][mask_tensor[0]].mean().item()
            else:
                last_val = 0
            
            buffer.finish_path(last_val)
            episode_returns.append(ep_ret)
            
            if done or truncated:
                obs, info = env.reset()
            ep_ret, ep_len = 0, 0
    
    return episode_returns


def evaluate_policy(env, actor_critic, num_episodes, device, args):
    """评估策略性能"""
    episode_returns = []
    episode_lengths = []
    success_rate = 0
    
    eval_pbar = tqdm(range(num_episodes), desc="策略评估", leave=False, disable=args.local_rank != 0)
    
    for _ in eval_pbar:
        obs, info = env.reset()
        ep_ret, ep_len = 0, 0
        
        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_tensor = torch.as_tensor(info.get('agent_mask', np.ones_like(obs[0], dtype=bool)), 
                                        dtype=torch.bool, device=device).unsqueeze(0)
            
            with torch.no_grad():
                logits, _ = actor_critic.policy(obs_tensor)
            
            # 使用sample_action采样动作
            from tools.path_formation import sample_action
            action_map = sample_action(
                logits.cpu(),  # 将logits移到CPU以匹配环境
                env.agent_positions, 
                env.temperature,
                env.get_feature(),
                action_choice="sample"
            )
            
            obs, reward, done, truncated, info = env.step(action_map)
            ep_ret += reward if isinstance(reward, (int, float)) else reward.sum()
            ep_len += 1
            
            if done or truncated:
                break
        
        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)
        if info.get('success', False):
            success_rate += 1
    
    success_rate /= num_episodes
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_rate
    }


def train_rl(args, env, actor_critic, device):
    """主要的RL训练循环"""
    # 初始化缓冲区
    obs, info = env.reset()
    obs_dim = obs.shape
    act_dim = args.action_dim
    
    # 中心化策略: 每个timestep只存储一次经验
    # 为了支持并行收集，缓冲区初始化在CPU上，训练时再移到GPU
    buffer = PPOBuffer(obs_dim, act_dim, args.steps_per_epoch, args.gamma, args.lam, 'cpu')
    trainer = PPOTrainer(actor_critic, args, device)
    
    # 训练统计
    episode_returns = deque(maxlen=100)
    best_mean_return = float('-inf')
    
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="RL训练进度", disable=args.local_rank != 0)
    
    for epoch in epoch_pbar:
        # 收集经验（选择并行或单线程）
        use_parallel = args.parallel_collect
        
        if use_parallel:
            ep_returns = collect_experience_parallel(env, actor_critic, buffer, args.steps_per_epoch, device, args)
        else:
            ep_returns = collect_experience(env, actor_critic, buffer, args.steps_per_epoch, device, args)
        
        episode_returns.extend(ep_returns)
        
        # 更新策略
        update_info = trainer.update(buffer)
        
        # 记录训练信息
        if args.local_rank == 0 and len(episode_returns) > 0:
            mean_return = np.mean(episode_returns)
            std_return = np.std(episode_returns)
            
            args.writer.add_scalar("RL/EpisodeReturn", mean_return, epoch)
            args.writer.add_scalar("RL/EpisodeReturnStd", std_return, epoch)
            args.writer.add_scalar("RL/PolicyLoss", update_info['pi_loss'], epoch)
            args.writer.add_scalar("RL/ValueLoss", update_info['v_loss'], epoch)
            args.writer.add_scalar("RL/KL", update_info['kl'], epoch)
            args.writer.add_scalar("RL/Entropy", update_info['entropy'], epoch)
            args.writer.add_scalar("RL/ClipFraction", update_info['clip_frac'], epoch)
            
            # 更新进度条描述
            epoch_pbar.set_postfix({
                'Return': f'{mean_return:.2f}±{std_return:.2f}',
                'PiLoss': f'{update_info["pi_loss"]:.4f}',
                'VLoss': f'{update_info["v_loss"]:.4f}',
                'KL': f'{update_info["kl"]:.4f}',
                'Entropy': f'{update_info["entropy"]:.4f}'
            })
        
        # 定期评估
        if epoch % args.eval_interval == 0:
            eval_results = evaluate_policy(env, actor_critic, args.num_eval_episodes, device, args)
            
            if args.local_rank == 0:
                args.writer.add_scalar("RL_Eval/MeanReturn", eval_results['mean_return'], epoch)
                args.writer.add_scalar("RL_Eval/MeanLength", eval_results['mean_length'], epoch)
                args.writer.add_scalar("RL_Eval/SuccessRate", eval_results['success_rate'], epoch)
                
                print(f"  Evaluation - Mean Return: {eval_results['mean_return']:.2f}")
                print(f"  Evaluation - Success Rate: {eval_results['success_rate']:.2f}")
                
                # 保存最佳模型
                if eval_results['mean_return'] > best_mean_return:
                    best_mean_return = eval_results['mean_return']
                    best_model_path = os.path.join(args.real_log_dir, "best_model.pth")
                    if args.distributed:
                        torch.save(actor_critic.module.state_dict(), best_model_path)
                    else:
                        torch.save(actor_critic.state_dict(), best_model_path)
                    print(f"  New best model saved! Return: {best_mean_return:.2f}")
        
        # 定期保存模型
        if epoch % args.save_interval == 0 and args.local_rank == 0:
            model_path = os.path.join(args.real_log_dir, f"model_epoch_{epoch}.pth")
            if args.distributed:
                torch.save(actor_critic.module.state_dict(), model_path)
            else:
                torch.save(actor_critic.state_dict(), model_path)
            print(f"  Model saved: {model_path}")


if __name__ == "__main__":
    # 获取强化学习专用参数
    args = get_rl_args()

    # 分布式训练设置
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        args.local_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda:{}".format(args.local_rank))
    else:
        args.local_rank = 0    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    args.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.real_log_dir = os.path.join(args.log_dir, f"RL_{args.current_time}")
    
    # 只在主进程创建tensorboard writer
    if args.local_rank == 0:
        args.writer = SummaryWriter(log_dir=args.real_log_dir)
        args_dict = vars(args)
        args_str = "\n".join([f"{key}: {value}" for key, value in args_dict.items()])
        args.writer.add_text("Args", args_str, 0)
        print("=== RL Training Configuration ===")
        print(args_str)
        print("================================")

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    if args.local_rank == 0:
        print("创建环境并计算距离图（仅计算一次，多线程共享）...")
    
    env = MAPFEnv(
        height=32,          # 地图高度
        width=32,           # 地图宽度  
        num_agents=args.num_agents,       # 代理数量
        obstacle_density=0.2, # 障碍物密度
        max_steps=args.max_episode_steps,      # 最大步数
        feature_dim=args.feature_dim,      # 特征维度
        feature_type=args.feature_type  # 特征类型
    )
    
    # 获取环境信息
    sample_obs, sample_info = env.reset()
    obs_dim = sample_obs.shape  # (channels, height, width)
    act_dim = args.action_dim   # 动作维度

    # 创建Actor-Critic网络
    actor_critic = PPOActorCritic(
        obs_dim, act_dim, 
        model_type=args.model,
        first_layer_channels=args.first_layer_channels,
        bilinear=args.bilinear
    )
    
    # 加载预训练模型（如果有）
    if args.model_path:
        print(f"Loading pretrained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 尝试加载策略网络权重
        actor_critic.policy.load_state_dict(checkpoint)
        print("Successfully loaded policy weights from supervised learning model")

    actor_critic.to(device)
    
    # 分布式训练包装
    if args.distributed:
        actor_critic = DDP(actor_critic, device_ids=[args.local_rank])

    # 打印模型信息
    if args.local_rank == 0:
        policy_params = sum(p.numel() for p in actor_critic.policy.parameters() if p.requires_grad)
        value_params = sum(p.numel() for p in actor_critic.value_net.parameters() if p.requires_grad)
        total_params = policy_params + value_params
        
        print(f"策略网络参数: {policy_params}")
        print(f"价值网络参数: {value_params}")
        print(f"总参数数量: {total_params}")
        print(f"模型大小约为: {total_params * 4 / (1024**2):.2f} MB")

    # 开始RL训练
    print("开始强化学习训练...")
    train_rl(args, env, actor_critic, device)
    
    print("RL训练完成！") 
