#!/usr/bin/env python3
"""
MAPF工具使用示例

展示如何使用新的C++加速工具进行距离地图预计算和路径转换
"""

import os
import sys
import time
from pathlib import Path

# 添加tools目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def example_distance_map_usage():
    """距离地图使用示例"""
    print("=== 距离地图使用示例 ===")
    
    # 1. 导入必要的模块
    from distance_map_reader import read_distance_map_cpp
    
    # 2. 查找一个地图文件
    data_dir = Path(__file__).parent.parent / "data"
    map_files_dir = data_dir / "map_files"
    
    if not map_files_dir.exists():
        print("❌ 数据目录不存在，请确保有测试数据")
        return False
    
    # 查找第一个.map文件
    map_files = list(map_files_dir.rglob("*.map"))
    if not map_files:
        print("❌ 未找到.map文件")
        return False
    
    map_file = str(map_files[0])
    print(f"使用地图文件: {Path(map_file).name}")
    
    try:
        # 3. 读取距离地图
        print("加载距离地图...")
        start_time = time.time()
        distance_map = read_distance_map_cpp(map_file)
        load_time = time.time() - start_time
        print(f"✅ 距离地图加载完成，耗时: {load_time:.3f}秒")
        
        # 4. 获取地图信息
        height, width = distance_map.get_map_size()
        valid_positions = distance_map.get_valid_positions()
        print(f"地图尺寸: {height}x{width}")
        print(f"有效位置数量: {len(valid_positions)}")
        
        # 5. 测试距离查询
        if len(valid_positions) >= 2:
            pos1 = valid_positions[0]
            pos2 = valid_positions[1]
            
            # 使用新的接口
            distance1 = distance_map.get_distance(pos1, pos2)
            print(f"从 {pos1} 到 {pos2} 的距离: {distance1}")
            
            # 测试多个距离查询
            if len(valid_positions) >= 5:
                test_positions = valid_positions[:5]
                print("测试多个位置间的距离:")
                for i, pos_a in enumerate(test_positions):
                    for j, pos_b in enumerate(test_positions[i+1:], i+1):
                        dist = distance_map.get_distance(pos_a, pos_b)
                        print(f"  {pos_a} -> {pos_b}: {dist}")
            
            print("✅ 距离查询测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def example_performance_test():
    """性能测试示例"""
    print("\n=== 性能测试示例 ===")
    
    try:
        # 运行简化的性能测试
        import subprocess
        benchmark_script = os.path.join(os.path.dirname(__file__), "benchmarks", "distance_benchmark.py")
        
        print("运行C++距离地图预计算性能测试...")
        result = subprocess.run([sys.executable, benchmark_script], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # 只显示总结部分
            lines = result.stdout.split('\n')
            summary_started = False
            for line in lines:
                if "性能测试总结" in line:
                    summary_started = True
                if summary_started and ("📊" in line or "💡" in line):
                    print(line)
            return True
        else:
            print(f"❌ 性能测试失败: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def example_integration_workflow():
    """完整工作流程示例"""
    print("\n=== 完整工作流程示例 ===")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # 检查数据目录
    if not (data_dir / "map_files").exists():
        print("❌ 缺少map_files目录")
        return False
    
    print("1. ✅ 地图文件目录存在")
    
    # 检查距离地图
    if not (data_dir / "distance_maps").exists():
        print("2. ❌ 距离地图未预计算，请运行:")
        print("   python mapf_tools.py distance -i data/map_files")
        return False
    
    print("2. ✅ 距离地图已预计算")
    
    # 检查路径文件
    if not (data_dir / "path_files").exists():
        print("3. ❌ 缺少path_files目录")
        return False
    
    print("3. ✅ 路径文件目录存在")
    
    # 检查二进制输入数据
    if not (data_dir / "input_data").exists():
        print("4. ❌ 二进制数据未生成，请运行:")
        print("   python mapf_tools.py convert -i data/path_files")
        return False
    
    print("4. ✅ 二进制输入数据已生成")
    
    print("\n🎉 工作流程完整，可以开始训练!")
    print("\n建议的完整流程:")
    print("1. python mapf_tools.py setup                    # 编译所有工具")
    print("2. python mapf_tools.py distance -i data/map_files  # 预计算距离地图")
    print("3. python mapf_tools.py convert -i data/path_files  # 转换路径文件")
    print("4. python train.py --dataset_path data/input_data   # 开始训练")
    
    return True

def main():
    """主函数"""
    print("MAPF工具使用示例")
    print("=" * 50)
    
    # 检查工具是否已编译
    tools_dir = Path(__file__).parent
    cpp_dir = tools_dir / "converters" / "cpp"
    
    required_tools = ["convert_lacam_path_to_bin", "precompute_distance_maps"]
    missing_tools = []
    
    for tool in required_tools:
        if not (cpp_dir / tool).exists():
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"❌ 缺少C++工具: {', '.join(missing_tools)}")
        print("请运行: python mapf_tools.py setup")
        return False
    
    print("✅ 所有C++工具已就绪")
    
    # 运行示例
    results = []
    
    # 距离地图使用示例
    results.append(example_distance_map_usage())
    
    # 完整工作流程示例
    results.append(example_integration_workflow())
    
    # 性能测试示例
    try:
        results.append(example_performance_test())
    except:
        print("⚠️ 性能测试跳过")
    
    # 总结
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 所有示例运行成功 ({passed}/{total})")
        return True
    else:
        print(f"⚠️ 部分示例失败 ({passed}/{total})")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 