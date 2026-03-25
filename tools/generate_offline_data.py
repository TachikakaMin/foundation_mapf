#!/usr/bin/env python
"""
并行生成离线训练数据。
Parallel offline training data generation.

用法 / Usage:
    python tools/generate_offline_data.py --config config.online.yaml --output_dir data/input_data --num_scenarios 100000 --workers 20

功能 / Features:
    - 调用 LACAM C++ 求解器生成 scenario
    - 多进程并行生成
    - 直接写入 .mbin 格式
    - 支持自定义地图和 agent 数量分布
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.extensions import generate_lacam_solution_cpp


INT32_MAX = 2_147_483_647


@dataclass
class PairConfig:
    """(地图, agent数量) 配置对"""
    map_file: str
    map_name: str
    density: int
    agent_num: int
    weight: int


@dataclass
class ScenarioData:
    """单个 scenario 的数据"""
    steps: int
    agent_num: int
    data: bytes  # 已序列化的二进制数据
    file_name: str


def extract_density_from_map_name(map_name: str) -> int:
    """从地图名称提取密度字段"""
    # Expected: maze-32-32-10-1-75-0  -> density field is 10.
    parts = map_name.split("-")
    if len(parts) < 4:
        return 0
    try:
        return int(parts[3])
    except ValueError:
        return 0


def offline_num_paths_weight(density: int, agent_num: int) -> int:
    """计算采样权重（与 MAPF_online_dataset.py 保持一致，但添加 512/1024）"""
    d = float(density)
    if agent_num == 1024:
        value = 5 + d * 0.2  # 低权重，因为很难求解
    elif agent_num == 512:
        value = 10 + d * 0.3
    elif agent_num == 256:
        value = 20 + d * 0.5
    elif agent_num == 128:
        value = 60 + d * 2.0
    elif agent_num == 96:
        value = 40 + d * 1.0
    elif agent_num == 64:
        value = 20 + d * 0.8
    elif agent_num == 32:
        value = 5 + d * 0.1
    elif agent_num == 16:
        value = 2 + d * 0.1
    else:
        return 0
    return int(math.floor(value + 0.5))


def build_pair_configs(
    map_files: Sequence[str], agent_counts: Sequence[int]
) -> List[PairConfig]:
    """构建 (地图, agent数量) 配置对列表"""
    pairs: List[PairConfig] = []
    for map_file in map_files:
        map_name = os.path.splitext(os.path.basename(map_file))[0]
        density = extract_density_from_map_name(map_name)
        for agent_num in agent_counts:
            weight = offline_num_paths_weight(density, agent_num)
            if weight > 0:
                pairs.append(
                    PairConfig(
                        map_file=map_file,
                        map_name=map_name,
                        density=density,
                        agent_num=int(agent_num),
                        weight=int(weight),
                    )
                )
    return pairs


def normalize_seed(raw_seed: int) -> int:
    """标准化 seed 到有效范围"""
    seed = int(raw_seed) % INT32_MAX
    if seed == 0:
        seed = 1
    return seed


def derive_actions_from_positions(positions: np.ndarray) -> np.ndarray:
    """从位置序列推导动作序列"""
    steps, agent_num, _ = positions.shape
    actions = np.zeros((steps, agent_num), dtype=np.uint8)
    if steps <= 1:
        return actions

    delta = positions[1:] - positions[:-1]
    dx = delta[:, :, 0]
    dy = delta[:, :, 1]

    step_actions = np.zeros_like(dx, dtype=np.uint8)
    step_actions[(dx == 0) & (dy == 1)] = 1   # 右
    step_actions[(dx == 0) & (dy == -1)] = 2  # 左
    step_actions[(dx == -1) & (dy == 0)] = 3  # 上
    step_actions[(dx == 1) & (dy == 0)] = 4   # 下
    actions[:-1] = step_actions
    return actions


def parse_lacam_output(raw) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """解析 LACAM 输出为 (positions, actions, goals)"""
    if raw is None:
        return None

    if isinstance(raw, dict):
        success = raw.get("success", raw.get("ok", True))
        if not success:
            return None

        positions = None
        for key in ("positions", "agent_positions", "agent_locations", "paths", "trajectory", "solution"):
            if key in raw:
                positions = raw[key]
                break

        actions = None
        for key in ("actions", "agent_actions", "action"):
            if key in raw:
                actions = raw[key]
                break

        goals = raw.get("goals", None)

    elif isinstance(raw, (tuple, list)):
        if len(raw) == 0:
            return None
        if len(raw) >= 3 and isinstance(raw[0], (bool, np.bool_)):
            if not bool(raw[0]):
                return None
            positions = raw[1]
            actions = raw[2]
            goals = raw[3] if len(raw) >= 4 else None
        elif len(raw) >= 2:
            positions = raw[0]
            actions = raw[1]
            goals = raw[2] if len(raw) >= 3 else None
        else:
            positions = raw[0]
            actions = None
            goals = None
    else:
        positions = raw
        actions = None
        goals = None

    if positions is None:
        return None

    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 3 or positions.shape[2] != 2:
        return None

    steps, agent_num, _ = positions.shape
    if steps <= 0 or agent_num <= 0:
        return None

    if goals is None:
        goals = positions[-1]
    else:
        goals = np.asarray(goals, dtype=np.int64)
        if goals.ndim != 2 or goals.shape != (agent_num, 2):
            return None

    if actions is None:
        actions = derive_actions_from_positions(positions)
    else:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.ndim == 3 and actions.shape[2] == 1:
            actions = actions[:, :, 0]
        if actions.ndim != 2 or actions.shape[1] != agent_num:
            return None
        if actions.shape[0] == steps - 1:
            padded = np.zeros((steps, agent_num), dtype=np.uint8)
            padded[:-1] = actions
            actions = padded
        elif actions.shape[0] != steps:
            return None

    return positions.astype(np.uint8), actions.astype(np.uint8), goals.astype(np.uint8)


def serialize_scenario(positions: np.ndarray, actions: np.ndarray) -> bytes:
    """将 scenario 序列化为 .mbin 格式的二进制数据"""
    steps, agent_num = positions.shape[:2]

    # 格式：steps(2) + agent_num(2) + [每个 timestep: positions(agent_num*2) + actions(agent_num)]
    data = bytearray()
    data.extend(struct.pack('<H', steps))
    data.extend(struct.pack('<H', agent_num))

    for t in range(steps):
        # 位置 (x, y) for each agent
        for agent_id in range(agent_num):
            data.append(positions[t, agent_id, 0])
            data.append(positions[t, agent_id, 1])
        # 动作 for each agent
        for agent_id in range(agent_num):
            data.append(actions[t, agent_id])

    return bytes(data)


def generate_one_scenario(
    pair: PairConfig,
    scenario_seed: int,
    time_limit_sec: int,
    verbose: int,
) -> Optional[ScenarioData]:
    """生成单个 scenario"""
    try:
        raw = generate_lacam_solution_cpp(
            map_file=pair.map_file,
            agent_num=int(pair.agent_num),
            seed=int(scenario_seed),
            time_limit_sec=int(time_limit_sec),
            verbose=int(verbose),
        )
    except RuntimeError:
        return None
    except Exception:
        return None

    parsed = parse_lacam_output(raw)
    if parsed is None:
        return None

    positions, actions, goals = parsed
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        return None

    data = serialize_scenario(positions, actions)
    file_name = f"{pair.map_name}-{pair.agent_num}-{scenario_seed}"

    return ScenarioData(
        steps=int(positions.shape[0]),
        agent_num=int(pair.agent_num),
        data=data,
        file_name=file_name,
    )


def worker_process(
    worker_id: int,
    pairs: List[PairConfig],
    pair_probs: np.ndarray,
    base_seed: int,
    time_limit_sec: int,
    verbose: int,
    retry_limit: int,
    num_scenarios: int,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    """Worker 进程：生成 scenario 并发送到队列"""
    rng = np.random.default_rng(base_seed + worker_id * 104_729)

    for _ in range(num_scenarios):
        success = False
        for _ in range(retry_limit):
            pair_idx = int(rng.choice(len(pairs), p=pair_probs))
            pair = pairs[pair_idx]
            scenario_seed = normalize_seed(base_seed + rng.integers(0, INT32_MAX))

            scenario = generate_one_scenario(pair, scenario_seed, time_limit_sec, verbose)
            if scenario is not None:
                result_queue.put((pair.map_name, pair.agent_num, scenario))
                progress_queue.put(1)
                success = True
                break

        if not success:
            progress_queue.put(0)  # 失败


def write_mbin_file(output_path: str, scenarios: List[ScenarioData]):
    """写入 .mbin 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        # 文件头部 (16字节)
        num_scenarios = len(scenarios)
        f.write(struct.pack('<I', num_scenarios))
        f.write(b'\x00' * 12)  # 预留空间

        # 计算偏移量
        header_size = 16
        index_table_size = num_scenarios * 272  # 每个索引 272 字节
        data_start_offset = header_size + index_table_size

        # 写入索引表
        current_offset = data_start_offset
        for scenario in scenarios:
            data_size = len(scenario.data)
            f.write(struct.pack('<Q', current_offset))  # offset (8 bytes)
            f.write(struct.pack('<I', data_size))       # data_size (4 bytes)
            f.write(struct.pack('<H', scenario.steps))  # steps (2 bytes)
            f.write(struct.pack('<H', scenario.agent_num))  # agent_num (2 bytes)

            # 文件名 (256 bytes)
            file_name_bytes = scenario.file_name.encode('utf-8')[:255]
            f.write(file_name_bytes + b'\x00' * (256 - len(file_name_bytes)))

            current_offset += data_size

        # 写入 scenario 数据
        for scenario in scenarios:
            f.write(scenario.data)


def main():
    parser = argparse.ArgumentParser(description="并行生成离线训练数据")
    parser.add_argument("--config", default="config.online.yaml", help="配置文件路径")
    parser.add_argument("--output_dir", default="data/input_data", help="输出目录")
    parser.add_argument("--num_scenarios", type=int, default=100000, help="生成的 scenario 总数")
    parser.add_argument("--workers", type=int, default=20, help="并行 worker 数量")
    parser.add_argument("--time_limit", type=int, default=5, help="LACAM 求解时间限制（秒）")
    parser.add_argument("--retry_limit", type=int, default=3, help="每个 scenario 的重试次数")
    parser.add_argument("--seed", type=int, default=1919180, help="随机种子")
    parser.add_argument("--agent_counts", type=int, nargs="+",
                        default=[16, 32, 64, 96, 128, 256, 512, 1024],
                        help="Agent 数量列表")
    parser.add_argument("--map_pattern", default="data/map_files/maze-*/*.map",
                        help="地图文件匹配模式")
    parser.add_argument("--scenarios_per_file", type=int, default=1000,
                        help="每个 .mbin 文件包含的 scenario 数量")
    args = parser.parse_args()

    # 加载地图文件
    map_files = sorted(glob.glob(args.map_pattern))
    if not map_files:
        print(f"❌ 未找到地图文件: {args.map_pattern}")
        return 1

    print(f"✓ 找到 {len(map_files)} 个地图文件")

    # 构建配置对
    pairs = build_pair_configs(map_files, args.agent_counts)
    if not pairs:
        print("❌ 没有有效的 (地图, agent数量) 配置对")
        return 1

    weights = np.array([pair.weight for pair in pairs], dtype=np.float64)
    pair_probs = weights / weights.sum()

    print(f"✓ 构建了 {len(pairs)} 个配置对")
    print(f"✓ Agent 数量分布: {args.agent_counts}")

    # 创建队列
    result_queue = mp.Queue(maxsize=args.workers * 10)
    progress_queue = mp.Queue()

    # 启动 worker 进程
    scenarios_per_worker = args.num_scenarios // args.workers
    processes = []

    print(f"\n🚀 启动 {args.workers} 个 worker 进程...")
    for worker_id in range(args.workers):
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                pairs,
                pair_probs,
                args.seed,
                args.time_limit,
                0,  # verbose
                args.retry_limit,
                scenarios_per_worker,
                result_queue,
                progress_queue,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # 收集结果并写入文件
    scenarios_by_group: Dict[Tuple[str, int], List[ScenarioData]] = defaultdict(list)
    total_generated = 0
    total_failed = 0
    start_time = time.time()

    print(f"📊 生成进度:")

    while total_generated + total_failed < args.num_scenarios:
        # 处理进度更新
        try:
            success = progress_queue.get(timeout=0.1)
            if success:
                total_generated += 1
            else:
                total_failed += 1

            if (total_generated + total_failed) % 100 == 0:
                elapsed = time.time() - start_time
                rate = total_generated / elapsed if elapsed > 0 else 0
                eta = (args.num_scenarios - total_generated) / rate if rate > 0 else 0
                print(f"\r  生成: {total_generated}/{args.num_scenarios} "
                      f"({100*total_generated/args.num_scenarios:.1f}%), "
                      f"失败: {total_failed}, "
                      f"速率: {rate:.1f} scenarios/s, "
                      f"ETA: {eta/60:.1f} min", end="", flush=True)
        except:
            pass

        # 处理结果
        try:
            map_name, agent_num, scenario = result_queue.get_nowait()
            key = (map_name, agent_num)
            scenarios_by_group[key].append(scenario)

            # 如果达到文件大小限制，写入文件
            if len(scenarios_by_group[key]) >= args.scenarios_per_file:
                output_path = os.path.join(
                    args.output_dir,
                    f"{map_name}-{agent_num}",
                    f"{map_name}-{agent_num}-{len(scenarios_by_group[key])}.mbin"
                )
                write_mbin_file(output_path, scenarios_by_group[key])
                print(f"\n  ✓ 写入 {output_path} ({len(scenarios_by_group[key])} scenarios)")
                scenarios_by_group[key] = []
        except:
            pass

    # 等待所有进程结束
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    # 写入剩余的 scenario
    print(f"\n\n📝 写入剩余数据...")
    for (map_name, agent_num), scenarios in scenarios_by_group.items():
        if scenarios:
            output_path = os.path.join(
                args.output_dir,
                f"{map_name}-{agent_num}",
                f"{map_name}-{agent_num}-final.mbin"
            )
            write_mbin_file(output_path, scenarios)
            print(f"  ✓ 写入 {output_path} ({len(scenarios)} scenarios)")

    elapsed = time.time() - start_time
    print(f"\n✅ 完成！")
    print(f"  总生成: {total_generated} scenarios")
    print(f"  总失败: {total_failed} scenarios")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"  平均速率: {total_generated/elapsed:.1f} scenarios/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
