#!/bin/bash

# 检查必要的依赖
if ! command -v parallel &> /dev/null; then
    echo "错误: 需要安装 GNU parallel"
    echo "安装命令: sudo apt-get install parallel"
    exit 1
fi

if ! command -v bc &> /dev/null; then
    echo "错误: 需要安装 bc"
    echo "安装命令: sudo apt-get install bc"
    exit 1
fi

# 检查LACAM二进制文件
if [ ! -f "data_generation_LACAM/lacam3/build/main" ]; then
    echo "错误: LACAM 二进制文件不存在, 请先编译"
    exit 1
fi

# 创建输出目录
mkdir -p data/path_files

# 获取CPU核心数, 设置并行度
CPU_CORES=$(nproc)
PARALLEL_JOBS=$((CPU_CORES * 2))  # 使用2倍CPU核心数
echo "检测到 $CPU_CORES 个CPU核心, 设置并行任务数: $PARALLEL_JOBS"

for map_file in data/map_files/maze-*/*.map; do
    # Extract the map pattern from the full path (e.g., maze-64-64-10-1-0.1)
    map_pattern=$(basename $(dirname "$map_file"))
    map_name=$(basename "$map_file" .map)
    density=$(echo "$map_name" | awk -F'-' '{print $4}')

    # Define the N values and corresponding path calculations
    for N in 128 96 64 32 16; do
        case $N in
            128) num_paths=$(echo "60 + $density * 2" | bc) ;;
            96) num_paths=$(echo "40 + $density * 1" | bc) ;;
            64) num_paths=$(echo "20 + $density * 0.8" | bc) ;;
            32) num_paths=$(echo "5 + $density * 0.1" | bc) ;;
            16) num_paths=$(echo "2 + $density * 0.1" | bc) ;;
        esac

        # Convert num_paths to an integer
        num_paths=$(printf "%.0f" "$num_paths")

        mkdir -p "data/path_files/${map_pattern}/${map_name}-${N}"
        for seed in $(seq 1 ${num_paths}); do
            output_file="data/path_files/${map_pattern}/${map_name}-${N}/${map_name}-${N}-${seed}.path"
            if [ ! -f "$output_file" ]; then
                echo "$map_pattern $map_name $N $seed"
            fi
        done
    done

done | parallel --jobs $PARALLEL_JOBS --progress --bar --eta --timeout 5 --colsep ' ' \
    'data_generation_LACAM/lacam3/build/main -m data/map_files/{1}/{2}.map -N {3} -s {4} -v 1 -o data/path_files/{1}/{2}-{3}/{2}-{3}-{4}.path'