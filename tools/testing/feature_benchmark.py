#!/usr/bin/env python3
"""
特征构建函数性能对比

对比Python版本和C++版本的construct_input_feature函数性能
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def create_test_data(map_size=(32, 32), num_agents=16, device='cpu'):
    """创建测试数据"""
    height, width = map_size
    
    # 创建地图数据
    map_data = torch.randint(0, 2, (height, width), dtype=torch.float32, device=device)
    
    # 创建智能体位置（确保在有效位置）
    valid_positions = []
    for i in range(height):
        for j in range(width):
            if map_data[i, j] == 0:  # 可通行位置
                valid_positions.append((i, j))
    
    if len(valid_positions) < num_agents * 2:
        # 如果有效位置不够，创建更多空位
        map_data.fill_(0)
        valid_positions = [(i, j) for i in range(height) for j in range(width)]
    
    # 随机选择智能体和目标位置
    selected_positions = np.random.choice(len(valid_positions), num_agents * 2, replace=False)
    
    agent_locations = torch.tensor([valid_positions[i] for i in selected_positions[:num_agents]], 
                                 dtype=torch.long, device=device)
    goal_locations = torch.tensor([valid_positions[i] for i in selected_positions[num_agents:]], 
                                dtype=torch.long, device=device)
    
    # 创建简化的距离地图
    distance_map = {}
    for pos in valid_positions:
        distances = torch.randint(1, 50, (height, width), dtype=torch.int32)
        distance_map[pos] = distances
    
    return map_data, agent_locations, goal_locations, distance_map

def benchmark_python_version(map_data, agent_locations, goal_locations, distance_map, 
                           feature_dim, feature_type, num_runs=10):
    """测试Python版本性能"""
    from tools.utils import construct_input_feature
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        result = construct_input_feature(
            map_data, agent_locations, goal_locations, distance_map,
            feature_dim, feature_type
        )
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'result_shape': result.shape
    }

def benchmark_optimized_version(map_data, agent_locations, goal_locations, distance_map, 
                              feature_dim, feature_type, num_runs=10):
    """测试优化版本性能"""
    try:
        from tools.fast_features import construct_input_feature
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            result = construct_input_feature(
                map_data, agent_locations, goal_locations, distance_map,
                feature_dim, feature_type
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'result_shape': result.shape,
            'available': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'available': False
        }

def run_benchmark_suite():
    """运行完整的benchmark测试"""
    print("特征构建函数性能测试")
    print("=" * 60)
    
    # 测试配置
    test_configs = [
        {"map_size": (32, 32), "num_agents": 8, "feature_dim": 3, "feature_type": "basic"},
        {"map_size": (32, 32), "num_agents": 16, "feature_dim": 4, "feature_type": "basic"},
        {"map_size": (32, 32), "num_agents": 32, "feature_dim": 6, "feature_type": "gradient"},
        {"map_size": (64, 64), "num_agents": 16, "feature_dim": 6, "feature_type": "gradient"},
        {"map_size": (32, 32), "num_agents": 64, "feature_dim": 6, "feature_type": "gradient"},
    ]
    
    # 测试设备
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        print(f"\n🔧 测试设备: {device.upper()}")
        print("-" * 40)
        
        for config in test_configs:
            print(f"\n测试配置: {config['map_size']}, {config['num_agents']}智能体, "
                  f"特征维度{config['feature_dim']}, {config['feature_type']}")
            
            # 创建测试数据
            try:
                map_data, agent_locations, goal_locations, distance_map = create_test_data(
                    config['map_size'], config['num_agents'], device
                )
                
                # 测试Python版本
                print("测试Python版本...")
                python_result = benchmark_python_version(
                    map_data, agent_locations, goal_locations, distance_map,
                    config['feature_dim'], config['feature_type'], num_runs=5
                )
                
                # 测试优化版本
                print("测试优化版本...")
                optimized_result = benchmark_optimized_version(
                    map_data, agent_locations, goal_locations, distance_map,
                    config['feature_dim'], config['feature_type'], num_runs=5
                )
                
                # 输出结果
                print(f"{'版本':<15} {'平均时间(ms)':<12} {'标准差(ms)':<12} {'状态':<10}")
                print("-" * 50)
                
                py_time_ms = python_result['avg_time'] * 1000
                py_std_ms = python_result['std_time'] * 1000
                print(f"{'Python':<15} {py_time_ms:<12.2f} {py_std_ms:<12.2f} {'成功':<10}")
                
                if optimized_result.get('available', False):
                    opt_time_ms = optimized_result['avg_time'] * 1000
                    opt_std_ms = optimized_result['std_time'] * 1000
                    print(f"{'优化版本':<15} {opt_time_ms:<12.2f} {opt_std_ms:<12.2f} {'成功':<10}")
                    
                    # 计算加速比
                    speedup = python_result['avg_time'] / optimized_result['avg_time']
                    improvement = (1 - optimized_result['avg_time'] / python_result['avg_time']) * 100
                    print(f"\n🚀 性能提升: {speedup:.2f}x 加速，{improvement:.1f}% 时间减少")
                else:
                    print(f"{'优化版本':<15} {'N/A':<12} {'N/A':<12} {'失败':<10}")
                    print(f"错误: {optimized_result.get('error', '未知错误')}")
                
            except Exception as e:
                print(f"❌ 测试配置失败: {e}")
                continue

def main():
    """主函数"""
    try:
        run_benchmark_suite()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main() 