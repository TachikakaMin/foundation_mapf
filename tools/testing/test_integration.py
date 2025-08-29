#!/usr/bin/env python3
"""
集成测试脚本：测试C++版本转换器的正确性
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import time

def create_test_path_file(test_dir, filename="test.path"):
    """创建一个测试用的.path文件"""
    test_content = """# 测试路径文件
solution=1
t=0:(5,5) (7,8) (10,12)
t=1:(5,6) (7,8) (10,13)
t=2:(5,7) (8,8) (11,13)
t=3:(6,7) (8,9) (11,14)
t=4:(6,8) (9,9) (12,14)
"""
    
    path_file = os.path.join(test_dir, "path_files", filename)
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    
    with open(path_file, 'w') as f:
        f.write(test_content)
    
    return path_file

def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=cwd,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "命令超时"
    except Exception as e:
        return False, "", str(e)

def get_tools_paths():
    """获取工具路径"""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # tools/
    cpp_dir = os.path.join(base_dir, "converters", "cpp")
    testing_dir = os.path.join(base_dir, "testing")
    
    return {
        "cpp_converter": os.path.join(cpp_dir, "convert_lacam_path_to_bin"),
        "cpp_dir": cpp_dir,
        "tester": os.path.join(testing_dir, "test_converter"),
        "testing_dir": testing_dir
    }

def test_converter_performance(test_dir, num_files=100):
    """测试转换器性能"""
    print(f"\n==== 性能测试 (创建{num_files}个测试文件) ====")
    
    paths = get_tools_paths()
    
    # 创建多个测试文件
    for i in range(num_files):
        create_test_path_file(test_dir, f"test_{i:03d}.path")
    
    # 测试C++版本
    print("测试C++版本性能...")
    start_time = time.time()
    success, stdout, stderr = run_command(
        f"./convert_lacam_path_to_bin {test_dir}",
        cwd=paths["cpp_dir"]
    )
    cpp_time = time.time() - start_time
    
    if success:
        print(f"✓ C++版本转换完成，耗时: {cpp_time:.2f}秒")
    else:
        print(f"✗ C++版本转换失败: {stderr}")
        return False
    
    return True

def test_single_file_conversion():
    """测试单个文件转换的正确性"""
    print("\n==== 单文件转换测试 ====")
    
    paths = get_tools_paths()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试文件
        test_file = create_test_path_file(temp_dir, "single_test.path")
        print(f"创建测试文件: {test_file}")
        
        # 运行C++转换器
        success, stdout, stderr = run_command(
            f"./convert_lacam_path_to_bin {temp_dir}",
            cwd=paths["cpp_dir"]
        )
        
        if not success:
            print(f"✗ 转换失败: {stderr}")
            return False
        
        print("✓ 转换完成")
        
        # 运行测试验证
        success, stdout, stderr = run_command(
            f"./test_converter {test_file}",
            cwd=paths["testing_dir"]
        )
        
        if success:
            print("✓ 验证通过")
            print(stdout)
            return True
        else:
            print(f"✗ 验证失败: {stderr}")
            return False

def test_directory_conversion():
    """测试目录转换的正确性"""
    print("\n==== 目录转换测试 ====")
    
    paths = get_tools_paths()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建多个测试文件
        test_files = []
        for i in range(5):
            test_file = create_test_path_file(temp_dir, f"dir_test_{i}.path")
            test_files.append(test_file)
        
        print(f"创建了 {len(test_files)} 个测试文件")
        
        # 运行C++转换器
        success, stdout, stderr = run_command(
            f"./convert_lacam_path_to_bin {temp_dir}",
            cwd=paths["cpp_dir"]
        )
        
        if not success:
            print(f"✗ 转换失败: {stderr}")
            return False
        
        print("✓ 批量转换完成")
        print(stdout)
        
        # 运行测试验证
        success, stdout, stderr = run_command(
            f"./test_converter {temp_dir}",
            cwd=paths["testing_dir"]
        )
        
        if success:
            print("✓ 批量验证通过")
            print(stdout)
            return True
        else:
            print(f"✗ 批量验证失败: {stderr}")
            return False

def check_prerequisites():
    """检查先决条件"""
    print("==== 检查先决条件 ====")
    
    paths = get_tools_paths()
    
    if not os.path.exists(paths["cpp_converter"]):
        print(f"✗ 转换器不存在: {paths['cpp_converter']}")
        print("请运行 'cd converters/cpp && make' 来编译程序")
        return False
    
    if not os.path.exists(paths["tester"]):
        print(f"✗ 测试工具不存在: {paths['tester']}")
        print("请运行 'cd converters/cpp && make' 来编译程序")
        return False
    
    print("✓ 转换器存在")
    print("✓ 测试工具存在")
    return True

def main():
    """主测试函数"""
    print("C++路径转换器集成测试")
    print("=" * 50)
    
    # 检查先决条件
    if not check_prerequisites():
        return False
    
    all_passed = True
    
    # 测试单文件转换
    if not test_single_file_conversion():
        all_passed = False
    
    # 测试目录转换
    if not test_directory_conversion():
        all_passed = False
    
    # 性能测试
    with tempfile.TemporaryDirectory() as temp_dir:
        if not test_converter_performance(temp_dir, num_files=50):
            all_passed = False
    
    # 输出最终结果
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有集成测试都通过了！")
        print("C++转换器工作正常，可以安全使用。")
        return True
    else:
        print("⚠️  有测试失败，请检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 