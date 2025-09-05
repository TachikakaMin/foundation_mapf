#!/usr/bin/env python3
"""
端到端训练性能测试

比较使用C++加速前后的完整训练性能
"""

import sys
import os
import time
import subprocess
import tempfile
from pathlib import Path

def run_training_benchmark(use_cpp=True, batch_size=16, num_batches=10):
    """运行训练性能测试"""
    
    # 临时禁用/启用C++扩展
    tools_dir = Path(__file__).parent.parent
    cpp_so_file = tools_dir / "construct_features_native.cpython-39-x86_64-linux-gnu.so"
    backup_file = tools_dir / "construct_features_native.cpython-39-x86_64-linux-gnu.so.backup"
    
    try:
        if not use_cpp and cpp_so_file.exists():
            # 临时移动C++扩展文件以强制使用Python版本
            cpp_so_file.rename(backup_file)
            print(f"🐍 强制使用Python版本（已备份C++扩展）")
        elif use_cpp and backup_file.exists():
            # 恢复C++扩展文件
            backup_file.rename(cpp_so_file)
            print(f"🚀 使用C++加速版本")
        elif use_cpp:
            print(f"🚀 使用C++加速版本")
        else:
            print(f"🐍 使用Python版本")
        
        # 创建一个限制数据集大小的临时训练脚本
        limited_train_script = f"""
import sys
sys.path.append('.')
from train import *

# 修改train.py中的数据集限制
original_train_code = '''
        # 限制文件数量以加快测试
        test_list = [files[i] for i in test_indices][:5]   # 只用5个文件测试
        train_list = [files[i] for i in train_indices][:20]  # 只用20个文件训练
'''

# 运行修改后的训练
if __name__ == "__main__":
    exec(open('train.py').read().replace(
        'test_list = [files[i] for i in test_indices][:10]',
        'test_list = [files[i] for i in test_indices][:3]'
    ).replace(
        'train_list = [files[i] for i in train_indices][:10]', 
        'train_list = [files[i] for i in train_indices][:10]'
    ))
"""
        
        # 写入临时脚本
        with open('temp_limited_train.py', 'w') as f:
            f.write(limited_train_script)
        
        # 运行训练脚本
        cmd = [
            sys.executable, "temp_limited_train.py",
            "--batch_size", str(batch_size),
            "--epochs", "1", 
            "--num_workers", "0",
            "--eval_interval", "999",  # 跳过评估
            "--save_interval", "999"   # 跳过保存
        ]
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            return {
                'success': True,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            return {
                'success': False,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
    finally:
        # 确保恢复C++扩展（如果被移动了）
        if backup_file.exists():
            backup_file.rename(cpp_so_file)

def parse_training_stats(stdout):
    """从训练输出中解析统计信息"""
    lines = stdout.split('\n')
    stats = {
        'total_files': 0,
        'total_steps': 0,
        'batch_count': 0,
        'epoch_time': 0
    }
    
    for line in lines:
        if 'merged files with total' in line:
            # Found 100 merged files with total 357936 steps
            parts = line.split()
            if 'total' in parts:
                idx = parts.index('total')
                if idx + 1 < len(parts):
                    stats['total_steps'] += int(parts[idx + 1])
        elif 'Epoch 1/1:' in line and '100%' in line:
            # Epoch 1/1: 100%|████████| 691/691 [01:23<00:00,  8.26it/s]
            if '[' in line and '<' in line:
                time_part = line.split('[')[1].split('<')[0]
                # 解析时间格式 mm:ss
                if ':' in time_part:
                    minutes, seconds = time_part.split(':')
                    stats['epoch_time'] = int(minutes) * 60 + int(seconds)
    
    return stats

def main():
    """主函数"""
    print("端到端训练性能测试")
    print("=" * 60)
    
    batch_size = 32
    
    print(f"测试配置: batch_size={batch_size}, 1个epoch")
    print("注意: 这个测试会运行实际的训练，可能需要几分钟")
    
    results = {}
    
    # 测试Python版本
    print(f"\n🐍 测试Python版本...")
    python_result = run_training_benchmark(use_cpp=False, batch_size=batch_size)
    results['python'] = python_result
    
    if python_result['success']:
        python_stats = parse_training_stats(python_result['stdout'])
        print(f"✅ Python版本完成，耗时: {python_result['execution_time']:.2f}秒")
        print(f"   处理步数: {python_stats['total_steps']:,}")
    else:
        print(f"❌ Python版本失败: {python_result['stderr'][:200]}...")
    
    # 测试C++版本
    print(f"\n🚀 测试C++版本...")
    cpp_result = run_training_benchmark(use_cpp=True, batch_size=batch_size)
    results['cpp'] = cpp_result
    
    if cpp_result['success']:
        cpp_stats = parse_training_stats(cpp_result['stdout'])
        print(f"✅ C++版本完成，耗时: {cpp_result['execution_time']:.2f}秒")
        print(f"   处理步数: {cpp_stats['total_steps']:,}")
    else:
        print(f"❌ C++版本失败: {cpp_result['stderr'][:200]}...")
    
    # 性能对比
    if python_result['success'] and cpp_result['success']:
        print(f"\n{'='*60}")
        print("性能对比结果")
        print(f"{'='*60}")
        
        py_time = python_result['execution_time']
        cpp_time = cpp_result['execution_time']
        speedup = py_time / cpp_time
        improvement = (py_time - cpp_time) / py_time * 100
        
        print(f"Python版本耗时: {py_time:.2f}秒")
        print(f"C++版本耗时:    {cpp_time:.2f}秒")
        print(f"")
        print(f"🚀 加速比: {speedup:.2f}x")
        print(f"📊 时间减少: {improvement:.1f}%")
        print(f"⏱️ 节省时间: {py_time - cpp_time:.2f}秒")
        
        if speedup > 1.5:
            print(f"\n🎉 C++加速效果显著！建议在生产环境中使用")
        elif speedup > 1.1:
            print(f"\n👍 C++加速有效果，建议使用")
        else:
            print(f"\n🤔 C++加速效果有限，可能需要进一步优化")
    
    return results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 