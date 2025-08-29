#!/usr/bin/env python3
"""
MAPF数据处理状态检查工具

检查地图文件、距离地图、路径文件和二进制数据的处理状态
"""

import os
import sys
from pathlib import Path
import glob

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def count_files_and_size(directory, pattern="*"):
    """统计目录中文件数量和总大小"""
    if not os.path.exists(directory):
        return 0, 0
    
    files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    file_count = len([f for f in files if os.path.isfile(f)])
    total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    
    return file_count, total_size

def check_data_status():
    """检查数据处理状态"""
    print("MAPF数据处理状态检查")
    print("=" * 60)
    
    # 获取数据目录路径
    tools_dir = Path(__file__).parent
    data_dir = tools_dir.parent / "data"
    
    if not data_dir.exists():
        print("❌ 数据目录不存在")
        return False
    
    print(f"数据目录: {data_dir}")
    print()
    
    # 检查各个子目录的状态
    directories = {
        "map_files": ("地图文件", "*.map"),
        "distance_maps": ("距离地图", "*.dmap"),
        "path_files": ("路径文件", "*.path"),
        "input_data": ("二进制数据", "*.bin")
    }
    
    status_summary = {}
    
    for dir_name, (display_name, pattern) in directories.items():
        dir_path = data_dir / dir_name
        
        if dir_path.exists():
            file_count, total_size = count_files_and_size(str(dir_path), pattern)
            status_summary[dir_name] = {
                "exists": True,
                "count": file_count,
                "size": total_size,
                "display_name": display_name
            }
            
            status = "✅" if file_count > 0 else "⚠️"
            print(f"{status} {display_name:<12}: {file_count:>8,} 个文件, {format_size(total_size):>10}")
        else:
            status_summary[dir_name] = {
                "exists": False,
                "count": 0,
                "size": 0,
                "display_name": display_name
            }
            print(f"❌ {display_name:<12}: 目录不存在")
    
    print()
    
    # 检查工具状态
    print("工具状态检查")
    print("-" * 30)
    
    cpp_dir = tools_dir / "converters" / "cpp"
    tools_status = {
        "convert_lacam_path_to_bin": "路径转换器",
        "precompute_distance_maps": "距离地图预计算",
        "../../testing/test_converter": "测试验证工具"
    }
    
    for tool_name, description in tools_status.items():
        tool_path = cpp_dir / tool_name
        if tool_path.exists():
            print(f"✅ {description:<16}: 已编译")
        else:
            print(f"❌ {description:<16}: 未编译")
    
    print()
    
    # 给出建议
    print("处理建议")
    print("-" * 30)
    
    if not status_summary["map_files"]["exists"] or status_summary["map_files"]["count"] == 0:
        print("1. ❌ 需要准备地图文件 (.map)")
    else:
        print("1. ✅ 地图文件已准备")
    
    if not status_summary["distance_maps"]["exists"] or status_summary["distance_maps"]["count"] == 0:
        print("2. 🔄 建议预计算距离地图:")
        print("   python mapf_tools.py distance -i ../data/map_files")
    else:
        map_count = status_summary["map_files"]["count"]
        dist_count = status_summary["distance_maps"]["count"]
        if dist_count >= map_count:
            print("2. ✅ 距离地图已完成")
        else:
            print(f"2. ⚠️ 距离地图不完整 ({dist_count}/{map_count})")
            print("   python mapf_tools.py distance -i ../data/map_files")
    
    if not status_summary["path_files"]["exists"] or status_summary["path_files"]["count"] == 0:
        print("3. ❌ 需要准备路径文件 (.path)")
    else:
        print("3. ✅ 路径文件已准备")
    
    if not status_summary["input_data"]["exists"] or status_summary["input_data"]["count"] == 0:
        print("4. 🔄 建议转换路径文件:")
        print("   python mapf_tools.py convert -i ../data/path_files")
    else:
        path_count = status_summary["path_files"]["count"]
        bin_count = status_summary["input_data"]["count"]
        if bin_count >= path_count:
            print("4. ✅ 二进制数据已完成")
        else:
            print(f"4. ⚠️ 二进制数据不完整 ({bin_count}/{path_count})")
            print("   python mapf_tools.py convert -i ../data/path_files")
    
    # 检查是否可以开始训练
    print()
    all_ready = (
        status_summary["map_files"]["count"] > 0 and
        status_summary["distance_maps"]["count"] > 0 and
        status_summary["input_data"]["count"] > 0
    )
    
    if all_ready:
        print("🎉 所有数据已准备完成，可以开始训练！")
        print("   python train.py --dataset_path data/input_data")
    else:
        print("⚠️ 数据准备未完成，请按照上述建议执行相应步骤")
    
    return all_ready

def main():
    """主函数"""
    try:
        return check_data_status()
    except Exception as e:
        print(f"❌ 状态检查失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 