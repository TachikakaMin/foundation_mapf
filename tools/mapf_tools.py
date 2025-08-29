#!/usr/bin/env python3
"""
MAPF工具集主入口脚本

提供统一的命令行界面来使用所有MAPF工具
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def get_script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

def run_converter(args):
    """运行路径转换器"""
    script_dir = get_script_dir()
    
    if args.engine == 'cpp':
        # 使用C++版本
        cpp_dir = os.path.join(script_dir, "converters", "cpp")
        converter = os.path.join(cpp_dir, "convert_lacam_path_to_bin")
        
        if not os.path.exists(converter):
            print("❌ C++转换器不存在，正在编译...")
            result = subprocess.run(["make"], cwd=cpp_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ 编译失败: {result.stderr}")
                return 1
            print("✅ 编译完成")
        
        # 转换为绝对路径
        abs_input_dir = os.path.abspath(args.input_dir)
        cmd = [converter, abs_input_dir]
        result = subprocess.run(cmd, cwd=cpp_dir)
        return result.returncode
        
    else:
        # 使用Python版本
        python_script = os.path.join(script_dir, "converters", "python", "convert_lacam_path_to_bin.py")
        cmd = [sys.executable, python_script, args.input_dir]
        result = subprocess.run(cmd)
        return result.returncode

def run_visualizer(args):
    """运行可视化工具"""
    script_dir = get_script_dir()
    vis_dir = os.path.join(script_dir, "visualization")
    
    if args.file.endswith('.path'):
        script = os.path.join(vis_dir, "visualize_lacam_path.py")
    elif args.file.endswith('.bin'):
        script = os.path.join(vis_dir, "visualize_bin_path.py")
    else:
        print("❌ 不支持的文件格式。支持: .path, .bin")
        return 1
    
    cmd = [sys.executable, script, args.file]
    if args.output:
        cmd.extend(["--output", args.output])
    
    result = subprocess.run(cmd)
    return result.returncode

def run_tester(args):
    """运行测试工具"""
    script_dir = get_script_dir()
    
    if args.type == 'unit':
        # 运行单元测试
        testing_dir = os.path.join(script_dir, "testing")
        tester = os.path.join(testing_dir, "test_converter")
        
        if not os.path.exists(tester):
            print("❌ 测试工具不存在，正在编译...")
            cpp_dir = os.path.join(script_dir, "converters", "cpp")
            result = subprocess.run(["make"], cwd=cpp_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ 编译失败: {result.stderr}")
                return 1
            print("✅ 编译完成")
        
        cmd = [tester]
        if args.target:
            cmd.append(args.target)
        
        result = subprocess.run(cmd, cwd=testing_dir)
        return result.returncode
        
    elif args.type == 'integration':
        # 运行集成测试
        test_script = os.path.join(script_dir, "testing", "test_integration.py")
        result = subprocess.run([sys.executable, test_script])
        return result.returncode
        
    elif args.type == 'benchmark':
        # 运行性能测试
        bench_script = os.path.join(script_dir, "benchmarks", "benchmark.py")
        result = subprocess.run([sys.executable, bench_script])
        return result.returncode

def run_distance_precompute(args):
    """运行距离地图预计算"""
    script_dir = get_script_dir()
    
    if args.engine == 'cpp':
        # 使用C++版本
        cpp_dir = os.path.join(script_dir, "converters", "cpp")
        distance_tool = os.path.join(cpp_dir, "precompute_distance_maps")
        
        if not os.path.exists(distance_tool):
            print("❌ C++距离地图工具不存在，正在编译...")
            result = subprocess.run(["make"], cwd=cpp_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ 编译失败: {result.stderr}")
                return 1
            print("✅ 编译完成")
        
        # 转换为绝对路径
        abs_input_dir = os.path.abspath(args.input_dir)
        cmd = [distance_tool, abs_input_dir]
        result = subprocess.run(cmd, cwd=cpp_dir)
        return result.returncode
        
    else:
        # 使用Python版本
        cmd = [sys.executable, "-m", "tools.precompute_distance_maps", args.input_dir]
        result = subprocess.run(cmd, cwd=os.path.dirname(script_dir))
        return result.returncode

def run_setup(args):
    """设置和编译所有工具"""
    script_dir = get_script_dir()
    cpp_dir = os.path.join(script_dir, "converters", "cpp")
    
    print("🔨 编译C++工具...")
    result = subprocess.run(["make", "clean"], cwd=cpp_dir, capture_output=True)
    result = subprocess.run(["make"], cwd=cpp_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 编译失败: {result.stderr}")
        return 1
    
    print("✅ 所有工具编译完成")
    
    # 运行测试验证
    print("🧪 运行验证测试...")
    test_script = os.path.join(script_dir, "testing", "test_integration.py")
    result = subprocess.run([sys.executable, test_script], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 所有测试通过，工具可以正常使用")
    else:
        print("⚠️ 有测试失败，请检查输出")
        print(result.stdout)
        print(result.stderr)
    
    return result.returncode

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MAPF工具集 - 多智能体路径规划工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s convert -i /path/to/lacam/results/           # 转换路径文件
  %(prog)s convert -i /path/to/lacam/results/ -e python # 使用Python版本
  %(prog)s distance -i /path/to/map/files/              # 预计算距离地图
  %(prog)s distance -i /path/to/map/files/ -e python    # 使用Python版本
  %(prog)s visualize /path/to/file.path                 # 可视化路径
  %(prog)s test unit /path/to/results/                  # 运行单元测试
  %(prog)s test integration                             # 运行集成测试
  %(prog)s test benchmark                               # 运行性能测试
  %(prog)s setup                                        # 编译和设置所有工具
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 转换命令
    convert_parser = subparsers.add_parser('convert', help='转换路径文件')
    convert_parser.add_argument('-i', '--input-dir', required=True, help='输入目录')
    convert_parser.add_argument('-e', '--engine', choices=['cpp', 'python'], 
                               default='cpp', help='转换引擎 (默认: cpp)')
    
    # 可视化命令
    vis_parser = subparsers.add_parser('visualize', help='可视化路径文件')
    vis_parser.add_argument('file', help='要可视化的文件')
    vis_parser.add_argument('-o', '--output', help='输出文件路径')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    test_parser.add_argument('type', choices=['unit', 'integration', 'benchmark'],
                            help='测试类型')
    test_parser.add_argument('target', nargs='?', help='测试目标 (用于单元测试)')
    
    # 设置命令
    setup_parser = subparsers.add_parser('setup', help='编译和设置所有工具')
    
    # 距离地图预计算命令
    distance_parser = subparsers.add_parser('distance', help='预计算距离地图')
    distance_parser.add_argument('-i', '--input-dir', required=True, help='地图文件目录')
    distance_parser.add_argument('-e', '--engine', choices=['cpp', 'python'], 
                               default='cpp', help='计算引擎 (默认: cpp)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'convert':
            return run_converter(args)
        elif args.command == 'visualize':
            return run_visualizer(args)
        elif args.command == 'test':
            return run_tester(args)
        elif args.command == 'setup':
            return run_setup(args)
        elif args.command == 'distance':
            return run_distance_precompute(args)
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 