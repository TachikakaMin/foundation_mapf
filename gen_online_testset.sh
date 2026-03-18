#!/bin/bash

set -euo pipefail

if ! command -v parallel &> /dev/null; then
    echo "错误: 需要安装 GNU parallel"
    echo "安装命令: sudo apt-get install parallel"
    exit 1
fi

if [ ! -f "data_generation_LACAM/lacam3/build/main" ]; then
    echo "错误: LACAM 二进制文件不存在, 请先编译"
    exit 1
fi

MAP_GLOB="${MAP_GLOB:-data/map_files/maze-*/*.map}"
MAX_MAPS_TOTAL="${MAX_MAPS_TOTAL:-8}"
AGENT_COUNTS="${AGENT_COUNTS:-128 96 64 32 16}"
SEEDS_PER_AGENT="${SEEDS_PER_AGENT:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"
PARALLEL_TIMEOUT_SEC="${PARALLEL_TIMEOUT_SEC:-5}"
LACAM_BIN="${LACAM_BIN:-data_generation_LACAM/lacam3/build/main}"
PATH_ROOT="${PATH_ROOT:-data/online_eval_path_files}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/online_eval_input_data}"

SELECTED_MAPS_FILE="$(mktemp)"
TASKS_FILE="$(mktemp)"
trap 'rm -f "$SELECTED_MAPS_FILE" "$TASKS_FILE"' EXIT

python - "$MAP_GLOB" "$MAX_MAPS_TOTAL" > "$SELECTED_MAPS_FILE" <<'PY'
import glob
import sys

map_glob = sys.argv[1]
max_maps = int(sys.argv[2])
map_files = sorted(glob.glob(map_glob))

if not map_files:
    sys.exit(0)

count = min(max_maps, len(map_files))
if count <= 0:
    sys.exit(0)

if count == len(map_files):
    selected = map_files
elif count == 1:
    selected = [map_files[0]]
else:
    indices = sorted(
        {
            round(i * (len(map_files) - 1) / (count - 1))
            for i in range(count)
        }
    )
    selected = [map_files[index] for index in indices]

for map_file in selected:
    print(map_file)
PY

if [ ! -s "$SELECTED_MAPS_FILE" ]; then
    echo "错误: 未找到地图文件, 请先运行 bash gen_mapfile.sh"
    exit 1
fi

SELECTED_MAP_COUNT="$(wc -l < "$SELECTED_MAPS_FILE" | tr -d ' ')"
echo "选中 $SELECTED_MAP_COUNT 张地图用于 online 固定验证/测试集"

mkdir -p "$PATH_ROOT" "$OUTPUT_ROOT"

python - "$SELECTED_MAPS_FILE" <<'PY'
import sys
from tools.precompute_distance_maps import process_single_map

with open(sys.argv[1], "r", encoding="utf-8") as f:
    map_files = [line.strip() for line in f if line.strip()]

for map_file in map_files:
    result = process_single_map(map_file)
    if result is None:
        print(f"跳过距离图: {map_file}")
    else:
        print(f"生成距离图: {map_file}")
PY

while IFS= read -r map_file; do
    map_pattern="$(basename "$(dirname "$map_file")")"
    map_name="$(basename "$map_file" .map)"
    for agent_num in $AGENT_COUNTS; do
        mkdir -p "$PATH_ROOT/$map_pattern/$map_name-$agent_num"
        for seed in $(seq 1 "$SEEDS_PER_AGENT"); do
            output_file="$PATH_ROOT/$map_pattern/$map_name-$agent_num/$map_name-$agent_num-$seed.path"
            if [ ! -f "$output_file" ]; then
                printf "%s\t%s\t%s\t%s\n" "$map_pattern" "$map_name" "$agent_num" "$seed" >> "$TASKS_FILE"
            fi
        done
    done
done < "$SELECTED_MAPS_FILE"

if [ -s "$TASKS_FILE" ]; then
    echo "开始生成在线验证/测试 .path 文件"
    parallel --jobs "$PARALLEL_JOBS" --progress --bar --eta --timeout "$PARALLEL_TIMEOUT_SEC" --colsep '\t' \
        "mkdir -p $PATH_ROOT/{1}/{2}-{3} && $LACAM_BIN -m data/map_files/{1}/{2}.map -N {3} -s {4} -v 1 -o $PATH_ROOT/{1}/{2}-{3}/{2}-{3}-{4}.path" \
        :::: "$TASKS_FILE"
else
    echo "所有在线验证/测试 .path 文件已存在, 跳过生成"
fi

python - "$PATH_ROOT" "$OUTPUT_ROOT" <<'PY'
import glob
import os
import struct
import sys

from tools.convert_lacam_path_to_bin import convert_path_to_scenario_data, group_path_files

path_root = sys.argv[1]
output_root = sys.argv[2]

path_files = glob.glob(os.path.join(path_root, "**/*.path"), recursive=True)
if not path_files:
    print(f"错误: {path_root} 下没有找到 .path 文件")
    sys.exit(1)

grouped_files = group_path_files(path_files)
generated = 0

for (map_name, agent_num), files in sorted(grouped_files.items()):
    map_parts = map_name.split("-")
    if len(map_parts) < 7:
        continue

    map_pattern = "-".join(map_parts[:6])
    output_dir = os.path.join(output_root, map_pattern)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{map_name}-{agent_num}.mbin")

    scenarios = []
    for file_path in sorted(files):
        scenario = convert_path_to_scenario_data(file_path)
        if scenario:
            scenarios.append(scenario)

    if not scenarios:
        continue

    with open(output_file, "wb") as f:
        num_scenarios = len(scenarios)
        f.write(struct.pack("I", num_scenarios))
        f.write(b"\x00" * 12)

        header_size = 16
        index_table_size = num_scenarios * 272
        current_offset = header_size + index_table_size

        for scenario in scenarios:
            data_size = len(scenario["data"])
            steps = scenario["steps"]
            scenario_agent_num = scenario["agent_num"]
            file_name = scenario["file_name"]

            index_data = struct.pack("Q", current_offset)
            index_data += struct.pack("I", data_size)
            index_data += struct.pack("H", steps)
            index_data += struct.pack("H", scenario_agent_num)

            file_name_bytes = file_name.encode("utf-8")[:255]
            file_name_padded = file_name_bytes + b"\x00" * (256 - len(file_name_bytes))
            index_data += file_name_padded

            f.write(index_data)
            current_offset += data_size

        for scenario in scenarios:
            f.write(scenario["data"])

    generated += 1
    print(f"✅ 生成 {output_file}, 包含 {len(scenarios)} 个 scenarios")

print(f"✅ online 固定验证/测试集生成完成, 共 {generated} 个 .mbin 文件")
PY

FIRST_SAMPLE="$(find "$OUTPUT_ROOT" -name '*.mbin' | sort | head -n 1)"
echo "验证/测试集目录: $OUTPUT_ROOT"
if [ -n "$FIRST_SAMPLE" ]; then
    echo "示例 sample_data_path: $FIRST_SAMPLE"
fi
