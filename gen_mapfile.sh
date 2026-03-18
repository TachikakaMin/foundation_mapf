#!/bin/bash

set -euo pipefail

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

HEIGHT="${HEIGHT:-32}"
WIDTH="${WIDTH:-32}"
DENSITY_START="${DENSITY_START:-0.1}"
DENSITY_END="${DENSITY_END:-0.2}"
DENSITY_STEP="${DENSITY_STEP:-0.1}"
COMPONENT_START="${COMPONENT_START:-1}"
COMPONENT_END="${COMPONENT_END:-10}"
GO_STRAIGHT_START="${GO_STRAIGHT_START:-0.75}"
GO_STRAIGHT_END="${GO_STRAIGHT_END:-0.85}"
GO_STRAIGHT_STEP="${GO_STRAIGHT_STEP:-0.05}"
CPU_CORES="$(nproc)"
RESERVED_CORES="${RESERVED_CORES:-2}"
DEFAULT_PARALLEL_JOBS=$((CPU_CORES - RESERVED_CORES))
if [ "$DEFAULT_PARALLEL_JOBS" -lt 1 ]; then
    DEFAULT_PARALLEL_JOBS=1
fi
PARALLEL_JOBS="${PARALLEL_JOBS:-$DEFAULT_PARALLEL_JOBS}"

mkdir -p data/map_files/

TASKS_FILE="$(mktemp)"
trap 'rm -f "$TASKS_FILE"' EXIT

for density in $(seq "$DENSITY_START" "$DENSITY_STEP" "$DENSITY_END"); do
    for component in $(seq "$COMPONENT_START" "$COMPONENT_END"); do
        for go_straight in $(seq "$GO_STRAIGHT_START" "$GO_STRAIGHT_STEP" "$GO_STRAIGHT_END"); do
            num_maps=$(printf "%.0f" "$(echo "12 + (${density} * 30) - (${component} * 2)" | bc)")
            printf "%s\t%s\t%s\t%s\n" "$density" "$component" "$go_straight" "$num_maps" >> "$TASKS_FILE"
        done
    done
done

TOTAL_TASKS="$(wc -l < "$TASKS_FILE" | tr -d ' ')"
echo "共 $TOTAL_TASKS 组地图生成任务"
echo "检测到 $CPU_CORES 个 CPU 核心, 预留核心数: $RESERVED_CORES, 设置并行任务数: $PARALLEL_JOBS"

parallel --jobs "$PARALLEL_JOBS" --progress --bar --eta --colsep '\t' \
    "python data_generation_LACAM/maze_generator.py --num_maps {4} --width $((WIDTH-2)) --height $((HEIGHT-2)) --obstacle_density {1} --wall_components {2} --go_straight {3}" \
    :::: "$TASKS_FILE"

echo "✅ 地图生成完成"
