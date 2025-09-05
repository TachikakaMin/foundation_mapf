#!/usr/bin/env python3
"""
性能对比脚本：比较Python和C++版本的转换器性能
"""

import os
import sys
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
import psutil
import threading
import statistics

def create_test_path_files(test_dir, num_files=100, complexity='medium'):
    """创建测试用的.path文件"""
    
    if complexity == 'simple':
        # 简单场景：少智能体，短路径
        agents = 3
        steps = 10
        coords_template = [(5, 5), (7, 8), (10, 12)]
    elif complexity == 'medium':
        # 中等场景：中等智能体，中等路径
        agents = 10
        steps = 50
        coords_template = [(i*2, i*3) for i in range(agents)]
    else:  # complex
        # 复杂场景：多智能体，长路径
        agents = 50
        steps = 200
        coords_template = [(i*2, i*3) for i in range(agents)]
    
    path_files = []
    for i in range(num_files):
        test_content = "# 测试路径文件\nsolution=1\n"
        
        for t in range(steps):
            coords_str = " ".join([f"({x+t},{y+t})" for x, y in coords_template[:agents]])
            test_content += f"t={t}:{coords_str}\n"
        
        path_file = os.path.join(test_dir, "path_files", f"test_{i:03d}.path")
        os.makedirs(os.path.dirname(path_file), exist_ok=True)
        
        with open(path_file, 'w') as f:
            f.write(test_content)
        
        path_files.append(path_file)
    
    return path_files

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.process = None
    
    def start_monitoring(self, process_name):
        """开始监控指定进程"""
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        
        def monitor():
            while self.monitoring:
                try:
                    # 查找进程
                    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                        if process_name in proc.info['name']:
                            self.cpu_usage.append(proc.info['cpu_percent'])
                            self.memory_usage.append(proc.info['memory_info'].rss / 1024 / 1024)  # MB
                            break
                    time.sleep(0.1)
                except:
                    pass
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        time.sleep(0.2)  # 确保监控线程结束
    
    def get_stats(self):
        """获取统计信息"""
        if not self.cpu_usage:
            return {"cpu_avg": 0, "cpu_max": 0, "mem_avg": 0, "mem_max": 0}
        
        return {
            "cpu_avg": statistics.mean(self.cpu_usage),
            "cpu_max": max(self.cpu_usage),
            "mem_avg": statistics.mean(self.memory_usage),
            "mem_max": max(self.memory_usage)
        }

def run_command_with_monitoring(cmd, cwd=None, monitor_process=None):
    """运行命令并监控性能"""
    monitor = PerformanceMonitor()
    
    if monitor_process:
        monitor.start_monitoring(monitor_process)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300  # 5分钟超时
        )
        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        success = False
        stdout = ""
        stderr = "命令超时"
    except Exception as e:
        success = False
        stdout = ""
        stderr = str(e)
    
    end_time = time.time()
    
    if monitor_process:
        monitor.stop_monitoring()
    
    execution_time = end_time - start_time
    performance_stats = monitor.get_stats()
    
    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "execution_time": execution_time,
        "performance": performance_stats
    }

def benchmark_python_version(test_dir):
    """测试Python版本性能"""
    print("测试Python版本...")
    
    # 清理之前的输出
    input_data_dir = test_dir.replace("path_files", "input_data")
    if os.path.exists(input_data_dir):
        shutil.rmtree(input_data_dir)
    
    # 运行Python版本
    cmd = f"python -m tools.convert_lacam_path_to_bin {test_dir}"
    result = run_command_with_monitoring(
        cmd,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        monitor_process="python"
    )
    
    return result

def benchmark_cpp_version(test_dir):
    """测试C++版本性能"""
    print("测试C++版本...")
    
    # 清理之前的输出
    input_data_dir = test_dir.replace("path_files", "input_data")
    if os.path.exists(input_data_dir):
        shutil.rmtree(input_data_dir)
    
    # 运行C++版本
    cmd = f"./convert_lacam_path_to_bin {test_dir}"
    result = run_command_with_monitoring(
        cmd,
        cwd=os.path.join(os.path.dirname(__file__), "../converters/cpp"),
        monitor_process="convert_lacam_path_to_bin"
    )
    
    return result

def run_benchmark(num_files, complexity='medium'):
    """运行性能测试"""
    print(f"\n==== 性能测试: {num_files}个文件, 复杂度={complexity} ====")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试文件
        print(f"创建 {num_files} 个测试文件...")
        test_files = create_test_path_files(temp_dir, num_files, complexity)
        
        # 计算总文件大小
        total_size = sum(os.path.getsize(f) for f in test_files) / 1024 / 1024  # MB
        print(f"总文件大小: {total_size:.2f} MB")
        
        # 测试Python版本
        python_result = benchmark_python_version(temp_dir)
        
        # 测试C++版本
        cpp_result = benchmark_cpp_version(temp_dir)
        
        # 输出结果
        print("\n--- 性能对比结果 ---")
        print(f"{'版本':<10} {'状态':<8} {'时间(秒)':<10} {'CPU平均%':<10} {'内存平均(MB)':<12} {'吞吐量(文件/秒)':<15}")
        print("-" * 80)
        
        if python_result["success"]:
            py_throughput = num_files / python_result["execution_time"]
            print(f"{'Python':<10} {'成功':<8} {python_result['execution_time']:<10.2f} "
                  f"{python_result['performance']['cpu_avg']:<10.1f} "
                  f"{python_result['performance']['mem_avg']:<12.1f} "
                  f"{py_throughput:<15.1f}")
        else:
            print(f"{'Python':<10} {'失败':<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<15}")
            print(f"错误: {python_result['stderr']}")
        
        if cpp_result["success"]:
            cpp_throughput = num_files / cpp_result["execution_time"]
            print(f"{'C++':<10} {'成功':<8} {cpp_result['execution_time']:<10.2f} "
                  f"{cpp_result['performance']['cpu_avg']:<10.1f} "
                  f"{cpp_result['performance']['mem_avg']:<12.1f} "
                  f"{cpp_throughput:<15.1f}")
            
            # 计算性能提升
            if python_result["success"]:
                speedup = python_result["execution_time"] / cpp_result["execution_time"]
                print(f"\n🚀 C++版本比Python版本快 {speedup:.2f} 倍")
        else:
            print(f"{'C++':<10} {'失败':<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<15}")
            print(f"错误: {cpp_result['stderr']}")
        
        return python_result, cpp_result

def main():
    """主函数"""
    print("路径转换器性能测试")
    print("=" * 60)
    
    # 检查程序是否存在
    tools_dir = os.path.dirname(os.path.dirname(__file__))
    cpp_converter = os.path.join(tools_dir, "converters", "cpp", "convert_lacam_path_to_bin")
    
    if not os.path.exists(cpp_converter):
        print("❌ C++转换器不存在，请先运行 'cd converters/cpp && make' 编译")
        return False
    
    # 运行不同规模的测试
    test_cases = [
        (10, 'simple'),
        (50, 'medium'),
        (100, 'medium'),
        (200, 'complex'),
    ]
    
    all_results = []
    
    for num_files, complexity in test_cases:
        try:
            python_result, cpp_result = run_benchmark(num_files, complexity)
            all_results.append((num_files, complexity, python_result, cpp_result))
        except KeyboardInterrupt:
            print("\n测试被用户中断")
            break
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    # 输出总结
    if all_results:
        print("\n" + "=" * 60)
        print("性能测试总结")
        print("=" * 60)
        
        speedups = []
        for num_files, complexity, py_result, cpp_result in all_results:
            if py_result["success"] and cpp_result["success"]:
                speedup = py_result["execution_time"] / cpp_result["execution_time"]
                speedups.append(speedup)
                print(f"{num_files}个{complexity}文件: {speedup:.2f}x 加速")
        
        if speedups:
            avg_speedup = statistics.mean(speedups)
            print(f"\n📊 平均加速比: {avg_speedup:.2f}x")
            print(f"📊 最大加速比: {max(speedups):.2f}x")
            print(f"📊 最小加速比: {min(speedups):.2f}x")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1) 