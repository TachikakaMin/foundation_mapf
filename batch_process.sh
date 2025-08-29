#!/bin/bash

# 分批处理脚本
# 将大量任务分批处理，避免内存和系统负载过高

BATCH_SIZE=1000  # 每批处理的任务数
PARALLEL_JOBS=48  # 并行任务数

echo "开始分批处理，批次大小: $BATCH_SIZE, 并行任务数: $PARALLEL_JOBS"

# 创建临时任务文件
TEMP_TASKS=$(mktemp)
trap "rm -f $TEMP_TASKS" EXIT

# 生成所有任务
echo "生成任务列表..."
for map_file in data/map_files/maze-*/*.map; do
    if [ -f "$map_file" ]; then
        map_pattern=$(basename $(dirname "$map_file"))
        map_name=$(basename "$map_file" .map)
        density=$(echo "$map_name" | awk -F'-' '{print $4}')

        for N in 128 96 64 32 16; do
            case $N in
                128) num_paths=$(echo "60 + $density * 2" | bc) ;;
                96) num_paths=$(echo "40 + $density * 1" | bc) ;;
                64) num_paths=$(echo "20 + $density * 0.8" | bc) ;;
                32) num_paths=$(echo "5 + $density * 0.1" | bc) ;;
                16) num_paths=$(echo "2 + $density * 0.1" | bc) ;;
            esac

            num_paths=$(printf "%.0f" "$num_paths")
            mkdir -p "data/path_files/${map_pattern}/${map_name}-${N}"
            
            for seed in $(seq 1 ${num_paths}); do
                output_file="data/path_files/${map_pattern}/${map_name}-${N}/${map_name}-${N}-${seed}.path"
                if [ ! -f "$output_file" ]; then
                    echo "$map_pattern $map_name $N $seed" >> "$TEMP_TASKS"
                fi
            done
        done
    fi
done

# 统计总任务数
TOTAL_TASKS=$(wc -l < "$TEMP_TASKS")
echo "总任务数: $TOTAL_TASKS"

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "所有任务已完成！"
    exit 0
fi

# 分批处理
BATCH_NUM=1
while [ -s "$TEMP_TASKS" ]; do
    echo "处理第 $BATCH_NUM 批..."
    
    # 提取当前批次的任务
    head -n "$BATCH_SIZE" "$TEMP_TASKS" > "${TEMP_TASKS}.batch"
    
    # 处理当前批次
    cat "${TEMP_TASKS}.batch" | parallel --jobs $PARALLEL_JOBS --progress --timeout 5 --colsep ' ' \
        'data_generation_LACAM/lacam3/build/main -m data/map_files/{1}/{2}.map -N {3} -s {4} -v 1 -o data/path_files/{1}/{2}-{3}/{2}-{3}-{4}.path'
    
    # 移除已处理的任务
    tail -n +$((BATCH_SIZE + 1)) "$TEMP_TASKS" > "${TEMP_TASKS}.remaining"
    mv "${TEMP_TASKS}.remaining" "$TEMP_TASKS"
    
    echo "第 $BATCH_NUM 批完成"
    ((BATCH_NUM++))
    
    # 短暂休息，避免系统过载
    sleep 1
done

echo "所有批次处理完成！" 