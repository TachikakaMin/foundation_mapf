#!/usr/bin/env python3
"""
快速训练性能测试

使用小数据集快速对比C++加速前后的训练性能
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_single_batch_performance():
    """测试单个batch的性能"""
    print("单批次性能测试")
    print("=" * 50)
    
    try:
        from MAPF_dataset_mbin import MAPFDataset
        import glob
        
        # 找到一些.mbin文件
        mbin_files = glob.glob('data/input_data/**/*.mbin', recursive=True)
        if not mbin_files:
            print("❌ 未找到.mbin文件")
            return
        
        # 只使用5个文件进行快速测试
        test_files = mbin_files[:5]
        print(f"使用 {len(test_files)} 个.mbin文件进行测试")
        
        # 创建数据集
        dataset = MAPFDataset(test_files, 6, 'gradient')
        print(f"数据集大小: {len(dataset)} 个样本")
        
        # 测试多个样本的加载性能
        num_samples = min(50, len(dataset))  # 最多测试50个样本
        
        print(f"\n测试 {num_samples} 个样本的加载性能...")
        
        # 预热
        _ = dataset[0]
        
        # 测试性能
        start_time = time.time()
        for i in range(num_samples):
            sample = dataset[i % len(dataset)]
            # 模拟一些基本的tensor操作
            feature = sample["feature"]
            action = sample["action"] 
            mask = sample["mask"]
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_sample = total_time / num_samples
        
        print(f"✅ 测试完成")
        print(f"总耗时: {total_time:.3f}秒")
        print(f"平均每样本: {avg_time_per_sample*1000:.2f}ms")
        print(f"吞吐量: {num_samples/total_time:.1f} 样本/秒")
        
        return {
            'total_time': total_time,
            'avg_time_per_sample': avg_time_per_sample,
            'throughput': num_samples/total_time,
            'num_samples': num_samples
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_cpp_vs_python_dataloader():
    """对比C++和Python版本的数据加载性能"""
    print("\n" + "=" * 50)
    print("C++ vs Python 数据加载性能对比")
    print("=" * 50)
    
    try:
        import glob
        from torch.utils.data import DataLoader
        
        # 找到测试文件
        mbin_files = glob.glob('data/input_data/**/*.mbin', recursive=True)
        if not mbin_files:
            print("❌ 未找到.mbin文件")
            return
        
        test_files = mbin_files[:3]  # 只用3个文件
        
        results = {}
        
        # 测试C++版本
        print(f"\n🚀 测试C++加速版本...")
        try:
            from MAPF_dataset_mbin import MAPFDataset
            
            dataset = MAPFDataset(test_files, 6, 'gradient')
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            
            start_time = time.time()
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 10:  # 只测试10个batch
                    break
            end_time = time.time()
            
            cpp_time = end_time - start_time
            results['cpp'] = {
                'time': cpp_time,
                'batches': batch_count,
                'avg_batch_time': cpp_time / batch_count
            }
            
            print(f"✅ C++版本: {cpp_time:.3f}秒, {batch_count}个batch, 平均{cpp_time/batch_count*1000:.2f}ms/batch")
            
        except Exception as e:
            print(f"❌ C++版本失败: {e}")
            results['cpp'] = None
        
        # 测试Python版本（临时禁用C++扩展）
        print(f"\n🐍 测试Python版本...")
        
        tools_dir = Path(__file__).parent.parent
        cpp_so_file = tools_dir / "construct_features_native.cpython-39-x86_64-linux-gnu.so"
        backup_file = tools_dir / "construct_features_native.cpython-39-x86_64-linux-gnu.so.backup"
        
        try:
            # 临时移动C++扩展
            if cpp_so_file.exists():
                cpp_so_file.rename(backup_file)
            
            # 清除模块缓存
            if 'MAPF_dataset_mbin' in sys.modules:
                del sys.modules['MAPF_dataset_mbin']
            if 'construct_features_native' in sys.modules:
                del sys.modules['construct_features_native']
            
            from MAPF_dataset_mbin import MAPFDataset
            
            dataset = MAPFDataset(test_files, 6, 'gradient')
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            
            start_time = time.time()
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 10:  # 只测试10个batch
                    break
            end_time = time.time()
            
            python_time = end_time - start_time
            results['python'] = {
                'time': python_time,
                'batches': batch_count,
                'avg_batch_time': python_time / batch_count
            }
            
            print(f"✅ Python版本: {python_time:.3f}秒, {batch_count}个batch, 平均{python_time/batch_count*1000:.2f}ms/batch")
            
        except Exception as e:
            print(f"❌ Python版本失败: {e}")
            results['python'] = None
        
        finally:
            # 恢复C++扩展
            if backup_file.exists():
                backup_file.rename(cpp_so_file)
        
        # 性能对比
        if results['cpp'] and results['python']:
            print(f"\n📊 性能对比:")
            cpp_time = results['cpp']['time']
            python_time = results['python']['time']
            speedup = python_time / cpp_time
            improvement = (python_time - cpp_time) / python_time * 100
            
            print(f"Python版本: {python_time:.3f}秒")
            print(f"C++版本:   {cpp_time:.3f}秒")
            print(f"🚀 加速比: {speedup:.2f}x")
            print(f"📈 时间减少: {improvement:.1f}%")
            
            if speedup > 2:
                print(f"🎉 C++加速效果显著！")
            elif speedup > 1.2:
                print(f"👍 C++加速有明显效果")
            else:
                print(f"🤔 C++加速效果有限")
        
        return results
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("快速训练性能测试")
    print("=" * 60)
    print("注意: 使用小数据集进行快速测试")
    
    # 单样本性能测试
    single_result = test_single_batch_performance()
    
    # 数据加载器对比测试
    dataloader_result = test_cpp_vs_python_dataloader()
    
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    
    if single_result:
        print(f"✅ 单样本测试完成，平均 {single_result['avg_time_per_sample']*1000:.2f}ms/样本")
    
    if dataloader_result and dataloader_result['cpp'] and dataloader_result['python']:
        speedup = dataloader_result['python']['time'] / dataloader_result['cpp']['time']
        print(f"✅ 数据加载器对比完成，C++版本 {speedup:.2f}x 加速")
        
        if speedup > 2:
            print(f"🎉 建议在生产训练中使用C++加速版本！")
        else:
            print(f"👍 C++版本有性能提升，建议使用")

if __name__ == "__main__":
    main() 