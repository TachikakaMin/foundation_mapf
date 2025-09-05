#!/usr/bin/env python3
"""
C++版本construct_input_feature性能测试

对比Python原版和C++版本的性能
"""

import sys
import os
import time
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def create_test_data(map_size=(32, 32), num_agents=16, device='cpu'):
    """创建测试数据"""
    height, width = map_size
    
    # 创建地图数据
    map_data = np.random.randint(0, 2, (height, width)).astype(np.float32)
    
    # 创建智能体位置
    agent_locations = np.random.randint(0, min(height, width), (num_agents, 2), dtype=np.int64)
    goal_locations = np.random.randint(0, min(height, width), (num_agents, 2), dtype=np.int64)
    
    # 创建简化的距离地图
    distance_map = {}
    for i in range(height):
        for j in range(width):
            if map_data[i, j] == 0:  # 可通行位置
                distances = np.random.randint(1, 50, (height, width), dtype=np.int32)
                distance_map[(i, j)] = distances
    
    # 转换为torch tensor
    if device != 'cpu':
        agent_locations = torch.from_numpy(agent_locations).to(device)
        goal_locations = torch.from_numpy(goal_locations).to(device)
    else:
        agent_locations = torch.from_numpy(agent_locations)
        goal_locations = torch.from_numpy(goal_locations)
    
    return map_data, agent_locations, goal_locations, distance_map

def benchmark_python_version(map_data, agent_locations, goal_locations, distance_map, 
                           feature_dim, feature_type, num_runs=10):
    """测试Python版本性能"""
    from tools.core.utils import construct_input_feature
    
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
        'result_shape': result.shape if hasattr(result, 'shape') else 'unknown'
    }

def benchmark_cpp_version(map_data, agent_locations, goal_locations, distance_map, 
                        feature_dim, feature_type, num_runs=10):
    """测试C++版本性能"""
    try:
        # 导入C++扩展
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        import construct_features_native as cpp_features
        
        times = []
        for _ in range(num_runs):
            # 转换为numpy数组
            map_data_np = map_data.astype(np.float32) if isinstance(map_data, np.ndarray) else np.array(map_data, dtype=np.float32)
            agent_locations_np = agent_locations.cpu().numpy().astype(np.int64) if hasattr(agent_locations, 'cpu') else agent_locations.astype(np.int64)
            goal_locations_np = goal_locations.cpu().numpy().astype(np.int64) if hasattr(goal_locations, 'cpu') else goal_locations.astype(np.int64)
            
            start_time = time.time()
            
            # 调用C++函数
            result_np = cpp_features.construct_input_feature(
                map_data_np,
                agent_locations_np,
                goal_locations_np,
                distance_map,
                feature_dim,
                feature_type
            )
            
            # 转换回torch tensor（如果需要）
            if hasattr(agent_locations, 'device'):
                result = torch.from_numpy(result_np).to(agent_locations.device)
            else:
                result = torch.from_numpy(result_np)
            
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

def run_comprehensive_benchmark():
    """运行全面的性能测试"""
    print("C++版本 construct_input_feature 性能测试")
    print("=" * 70)
    
    # 测试配置
    test_configs = [
        # 简单场景
        {"map_size": (32, 32), "num_agents": 8, "feature_dim": 3, "feature_type": "basic"},
        {"map_size": (32, 32), "num_agents": 16, "feature_dim": 4, "feature_type": "basic"},
        
        # 中等复杂度
        {"map_size": (32, 32), "num_agents": 32, "feature_dim": 6, "feature_type": "gradient"},
        {"map_size": (64, 64), "num_agents": 16, "feature_dim": 6, "feature_type": "gradient"},
        
        # 复杂场景
        {"map_size": (32, 32), "num_agents": 64, "feature_dim": 6, "feature_type": "gradient"},
        {"map_size": (64, 64), "num_agents": 32, "feature_dim": 6, "feature_type": "gradient"},
    ]
    
    # 测试设备
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    overall_results = []
    
    for device in devices:
        print(f"\n🔧 测试设备: {device.upper()}")
        print("-" * 50)
        
        for i, config in enumerate(test_configs):
            print(f"\n测试 {i+1}/{len(test_configs)}: {config['map_size']}, "
                  f"{config['num_agents']}智能体, 特征维度{config['feature_dim']}, {config['feature_type']}")
            
            try:
                # 创建测试数据
                map_data, agent_locations, goal_locations, distance_map = create_test_data(
                    config['map_size'], config['num_agents'], device
                )
                
                # 测试Python版本
                print("  测试Python版本...")
                python_result = benchmark_python_version(
                    map_data, agent_locations, goal_locations, distance_map,
                    config['feature_dim'], config['feature_type'], num_runs=5
                )
                
                # 测试C++版本
                print("  测试C++版本...")
                cpp_result = benchmark_cpp_version(
                    map_data, agent_locations, goal_locations, distance_map,
                    config['feature_dim'], config['feature_type'], num_runs=5
                )
                
                # 输出结果
                print(f"  {'版本':<12} {'平均时间(ms)':<15} {'标准差(ms)':<12} {'状态':<10}")
                print("  " + "-" * 55)
                
                py_time_ms = python_result['avg_time'] * 1000
                py_std_ms = python_result['std_time'] * 1000
                print(f"  {'Python':<12} {py_time_ms:<15.2f} {py_std_ms:<12.2f} {'成功':<10}")
                
                if cpp_result.get('available', False):
                    cpp_time_ms = cpp_result['avg_time'] * 1000
                    cpp_std_ms = cpp_result['std_time'] * 1000
                    print(f"  {'C++':<12} {cpp_time_ms:<15.2f} {cpp_std_ms:<12.2f} {'成功':<10}")
                    
                    # 计算性能提升
                    speedup = python_result['avg_time'] / cpp_result['avg_time']
                    improvement = (1 - cpp_result['avg_time'] / python_result['avg_time']) * 100
                    
                    status = "🚀" if speedup > 1.1 else "📊" if speedup > 0.9 else "⚠️"
                    print(f"  {status} 性能: {speedup:.2f}x 加速，{improvement:.1f}% 时间减少")
                    
                    overall_results.append({
                        'config': config,
                        'device': device,
                        'speedup': speedup,
                        'python_time': py_time_ms,
                        'cpp_time': cpp_time_ms
                    })
                else:
                    print(f"  {'C++':<12} {'N/A':<15} {'N/A':<12} {'失败':<10}")
                    print(f"  错误: {cpp_result.get('error', '未知错误')}")
                
            except Exception as e:
                print(f"  ❌ 测试配置失败: {e}")
                continue
    
    # 总结报告
    if overall_results:
        print(f"\n{'='*70}")
        print("性能测试总结")
        print(f"{'='*70}")
        
        speedups = [r['speedup'] for r in overall_results]
        avg_speedup = np.mean(speedups)
        
        print(f"📊 平均加速比: {avg_speedup:.2f}x")
        print(f"📊 最大加速比: {max(speedups):.2f}x")
        print(f"📊 最小加速比: {min(speedups):.2f}x")
        
        # 分类统计
        fast_cases = [r for r in overall_results if r['speedup'] > 1.1]
        slow_cases = [r for r in overall_results if r['speedup'] < 0.9]
        
        print(f"\n🚀 C++更快的场景: {len(fast_cases)}/{len(overall_results)}")
        for case in fast_cases[:3]:  # 显示前3个
            config = case['config']
            print(f"   - {config['map_size']}, {config['num_agents']}智能体: {case['speedup']:.2f}x")
        
        if slow_cases:
            print(f"\n⚠️ C++较慢的场景: {len(slow_cases)}/{len(overall_results)}")
            for case in slow_cases[:3]:  # 显示前3个
                config = case['config']
                print(f"   - {config['map_size']}, {config['num_agents']}智能体: {case['speedup']:.2f}x")
        
        print(f"\n💡 结论: C++版本在{len(fast_cases)}个场景中表现更好，平均加速{avg_speedup:.2f}x")

if __name__ == "__main__":
    run_comprehensive_benchmark() 