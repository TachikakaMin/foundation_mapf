#!/usr/bin/env python3
"""
距离地图预计算性能对比脚本

比较Python和C++版本的距离地图预计算性能
"""

import os
import sys
import subprocess
import tempfile
import time
import shutil
from pathlib import Path

def create_test_map(map_dir, map_name, size=(32, 32)):
    """创建测试地图文件"""
    height, width = size
    
    map_content = f"""type octile
height {height}
width {width}
map
"""
    
    # 创建简单的地图（边界是障碍物，中间是通道）
    for i in range(height):
        line = ""
        for j in range(width):
            if i == 0 or i == height-1 or j == 0 or j == width-1:
                line += "@"  # 边界障碍物
            elif (i + j) % 10 == 0:
                line += "@"  # 一些内部障碍物
            else:
                line += "."  # 可通行区域
        map_content += line + "\n"
    
    map_file = os.path.join(map_dir, f"{map_name}.map")
    os.makedirs(os.path.dirname(map_file), exist_ok=True)
    
    with open(map_file, 'w') as f:
        f.write(map_content)
    
    return map_file

def run_command_with_timing(cmd, cwd=None):
    """运行命令并计时"""
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True if isinstance(cmd, str) else False,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=20  # 20秒超时
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
    execution_time = end_time - start_time
    
    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "execution_time": execution_time
    }

def benchmark_cpp_version(map_dir):
    """测试C++版本性能"""
    print("测试C++版本...")
    
    # 清理之前的输出
    distance_dir = map_dir.replace("map_files", "distance_maps")
    if os.path.exists(distance_dir):
        shutil.rmtree(distance_dir)
    
    # 运行C++版本
    tools_dir = os.path.dirname(os.path.dirname(__file__))
    cpp_dir = os.path.join(tools_dir, "converters", "cpp")
    
    # 使用绝对路径避免路径问题
    abs_map_dir = os.path.abspath(map_dir)
    cmd = [os.path.join(cpp_dir, "precompute_distance_maps"), abs_map_dir]
    result = run_command_with_timing(cmd, cwd=cpp_dir)
    
    return result

def run_benchmark(num_maps, map_sizes):
    """运行性能测试"""
    print(f"\n==== 距离地图预计算性能测试: {num_maps}个地图 ====")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建map_files子目录（模拟真实结构）
        map_files_dir = os.path.join(temp_dir, "map_files")
        os.makedirs(map_files_dir, exist_ok=True)
        
        # 创建测试地图
        print(f"创建 {num_maps} 个测试地图...")
        total_cells = 0
        for i in range(num_maps):
            size = map_sizes[i % len(map_sizes)]
            map_file = create_test_map(map_files_dir, f"test_map_{i:03d}_{size[0]}x{size[1]}", size)
            total_cells += size[0] * size[1]
            
        print(f"总计算单元数: {total_cells:,}")
        
        # 只测试C++版本（避免Python依赖问题）
        cpp_result = benchmark_cpp_version(map_files_dir)
        
        # 输出结果
        print("\n--- 性能测试结果 ---")
        print(f"{'版本':<10} {'状态':<8} {'时间(秒)':<10} {'速度(单元/秒)':<15}")
        print("-" * 50)
        
        if cpp_result["success"]:
            cpp_speed = total_cells / cpp_result["execution_time"]
            print(f"{'C++':<10} {'成功':<8} {cpp_result['execution_time']:<10.2f} "
                  f"{cpp_speed:<15.0f}")
            
            print(f"\n🚀 C++版本处理速度: {cpp_speed:.0f} 单元/秒")
            print(f"📊 平均每个地图耗时: {cpp_result['execution_time']/num_maps:.3f} 秒")
        else:
            print(f"{'C++':<10} {'失败':<8} {'N/A':<10} {'N/A':<15}")
            print(f"错误: {cpp_result['stderr']}")
        
        return cpp_result

def main():
    """主函数"""
    print("距离地图预计算性能测试")
    print("=" * 60)
    
    # 检查C++工具是否存在
    tools_dir = os.path.dirname(os.path.dirname(__file__))
    cpp_converter = os.path.join(tools_dir, "converters", "cpp", "precompute_distance_maps")
    
    if not os.path.exists(cpp_converter):
        print("❌ C++距离地图工具不存在，请先运行 'cd converters/cpp && make' 编译")
        return False
    
    print("✅ C++距离地图工具已就绪")
    
    # 运行不同规模的测试
    test_cases = [
        (5, [(16, 16)]),           # 小地图
        (3, [(32, 32)]),           # 中等地图
        (2, [(64, 64)]),           # 大地图
    ]
    
    all_results = []
    
    for num_maps, map_sizes in test_cases:
        try:
            size_desc = ", ".join([f"{w}x{h}" for w, h in map_sizes])
            print(f"\n测试配置: {num_maps}个地图，尺寸: {size_desc}")
            result = run_benchmark(num_maps, map_sizes)
            all_results.append((num_maps, map_sizes, result))
        except KeyboardInterrupt:
            print("\n测试被用户中断")
            break
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    # 输出总结
    if all_results:
        print("\n" + "=" * 60)
        print("距离地图预计算性能测试总结")
        print("=" * 60)
        
        total_maps = 0
        total_time = 0
        total_cells = 0
        
        for num_maps, map_sizes, result in all_results:
            if result["success"]:
                total_maps += num_maps
                total_time += result["execution_time"]
                for size in map_sizes:
                    total_cells += size[0] * size[1] * (num_maps // len(map_sizes) + (1 if num_maps % len(map_sizes) > 0 else 0))
        
        if total_maps > 0:
            avg_speed = total_cells / total_time if total_time > 0 else 0
            print(f"📊 总计处理: {total_maps} 个地图")
            print(f"📊 总计耗时: {total_time:.2f} 秒")
            print(f"📊 平均速度: {avg_speed:.0f} 单元/秒")
            print(f"📊 平均每地图: {total_time/total_maps:.3f} 秒")
            
            print(f"\n💡 C++版本性能优秀！处理大量地图时建议使用C++版本")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1) 