#!/usr/bin/env python
"""
RAILGUN Scaling Law 实验脚本（统一版本）。
Scaling law experiment runner (Unified version).

用法 / Usage:
    # Online 模式
    python scaling_law.py --config config.online.yaml [--dry_run]

    # Offline 模式
    python scaling_law.py --config config.offline.yaml [--dry_run]

实验设计 / Experiment design:
    - N 轴（模型大小）: 5 个配置，通过 first_layer_channels × blocks_per_stage 控制
    - D 轴（数据量）: 1 次长训练中的多个 milestone step
    - 每个模型只训练一次到最大 step，并在 milestone 上记录 train/val/inference 指标
    - 拟合目标: L(N, D) = E + A/N^alpha + B/D^beta

模式切换 / Mode switching:
    - 通过 --config 参数自动检测模式
    - config.online.yaml → online 模式
    - config.offline.yaml → offline 模式
    - 自动使用对应的参数名称和数据路径
"""

import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime


# ── 实验 Grid 定义 ──────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    # (label, first_layer_channels, blocks_per_stage)
    # 固定宽度 flc=64，只在深度 (blocks_per_stage) 上缩放
    ("XS", 64, 0),   # 31.0M params — legacy DoubleConv
    ("S", 64, 1),    # 46.7M params
    ("M", 64, 2),    # 78.2M params
    ("L", 64, 3),    # 109.6M params
    ("XL", 64, 4),   # 141.0M params
]

STEP_CONFIGS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000,
                55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

# Online 模式的测试样本
ONLINE_INFERENCE_SAMPLES = [
    # 32x32 地图, 128 agents — 4 个不同场景
    "data/online_eval_input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-128.mbin",
    "data/online_eval_input_data/maze-32-32-10-2-80/maze-32-32-10-2-80-10-128.mbin",
    "data/online_eval_input_data/maze-32-32-20-1-75/maze-32-32-20-1-75-2-128.mbin",
    "data/online_eval_input_data/maze-32-32-20-5-75/maze-32-32-20-5-75-7-128.mbin",
    # 64x64 地图, 256 agents — 4 个不同场景
    "data/online_eval_input_data/maze-64-64-10-1-75/maze-64-64-10-1-75-0-256.mbin",
    "data/online_eval_input_data/maze-64-64-10-2-80/maze-64-64-10-2-80-1-256.mbin",
    "data/online_eval_input_data/maze-64-64-20-1-75/maze-64-64-20-1-75-14-256.mbin",
    "data/online_eval_input_data/maze-64-64-20-3-80/maze-64-64-20-3-80-6-256.mbin",
    # 64x64 地图, 512 agents — 2 个场景 (高密度)
    "data/online_eval_input_data/maze-64-64-10-1-75/maze-64-64-10-1-75-0-512.mbin",
    "data/online_eval_input_data/maze-64-64-20-2-75/maze-64-64-20-2-75-4-512.mbin",
    # 64x64 地图, 1024 agents — 2 个场景 (OOD 极端密度)
    "data/online_eval_input_data/maze-64-64-10-1-75/maze-64-64-10-1-75-0-1024.mbin",
    "data/online_eval_input_data/maze-64-64-10-2-80/maze-64-64-10-2-80-1-1024.mbin",
]

# Offline 模式的测试样本
OFFLINE_INFERENCE_SAMPLES = [
    # 32x32 地图, 128 agents — 4 个不同场景
    "data/input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-128.mbin",
    "data/input_data/maze-32-32-10-2-80/maze-32-32-10-2-80-10-128.mbin",
    "data/input_data/maze-32-32-20-1-75/maze-32-32-20-1-75-2-128.mbin",
    "data/input_data/maze-32-32-20-5-75/maze-32-32-20-5-75-7-128.mbin",
    # 64x64 地图, 256 agents — 4 个不同场景
    "data/input_data/maze-64-64-10-1-75/maze-64-64-10-1-75-0-256.mbin",
    "data/input_data/maze-64-64-10-2-80/maze-64-64-10-2-80-1-256.mbin",
    "data/input_data/maze-64-64-20-1-75/maze-64-64-20-1-75-14-256.mbin",
    "data/input_data/maze-64-64-20-3-80/maze-64-64-20-3-80-6-256.mbin",
    # 64x64 地图, 512 agents — 2 个场景 (高密度)
    "data/input_data/maze-64-64-10-1-75/maze-64-64-10-1-75-0-512.mbin",
    "data/input_data/maze-64-64-20-2-75/maze-64-64-20-2-75-4-512.mbin",
    # 64x64 地图, 1024 agents — 2 个场景 (OOD 极端密度)
    "data/input_data/maze-64-64-10-1-75/maze-64-64-10-1-75-0-1024.mbin",
    "data/input_data/maze-64-64-10-2-80/maze-64-64-10-2-80-1-1024.mbin",
]


STEP_PREFIX_RE = re.compile(r"^Step\s+(\d+)/(\d+),\s+")
TRAIN_LOSS_RE = re.compile(r"^Step\s+(\d+)/(\d+),\s+Training mean Loss:\s+([0-9eE+\-.]+)\s*$")
VAL_DIM_RE = re.compile(
    r"^Step\s+(\d+)/(\d+),\s+Validation mean Loss \[(\d+)x(\d+)\]:\s+([0-9eE+\-.]+)\s*$"
)
VAL_AGG_RE = re.compile(
    r"^Step\s+(\d+)/(\d+),\s+Validation mean Loss \(aggregated\):\s+([0-9eE+\-.]+)\s*$"
)
INFER_CASE_RE = re.compile(
    r"^Inference case\s+(\d+):\s+([^|]+)\|\s*(.+)\s*$"
)
INFER_SUMMARY_RE = re.compile(r"^Step\s+(\d+)/(\d+),\s+Inference summary:\s+(.+)\s*$")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _append_arg(cmd, key, value):
    if isinstance(value, (list, tuple)):
        cmd.append(f"--{key}")
        cmd.extend(str(item) for item in value)
        return
    cmd.extend([f"--{key}", str(value)])


def _sanitize_name(raw):
    return re.sub(r"[^0-9A-Za-z]+", "_", raw).strip("_").lower()


def _compute_event_interval_steps(milestone_steps):
    milestone_steps = [int(step) for step in milestone_steps]
    interval = milestone_steps[0]
    for step in milestone_steps[1:]:
        interval = math.gcd(interval, step)
    return max(1, int(interval))


def _resolve_inference_samples(samples, search_root):
    resolved = []
    for path in samples:
        if os.path.isfile(path):
            resolved.append(path)
            continue

        stem = os.path.splitext(os.path.basename(path))[0]
        candidates = glob.glob(os.path.join(search_root, "**", f"{stem}.mbin"), recursive=True)
        if not candidates:
            candidates = glob.glob(
                os.path.join(search_root, "**", f"{stem}-final.mbin"),
                recursive=True,
            )
        if candidates:
            resolved.append(sorted(candidates)[0])
    return resolved


def _python_has_torch(python_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [python_bin, "-c", "import torch"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0


def detect_mode(config_path: str) -> str:
    """从配置文件路径检测训练模式（online 或 offline）"""
    config_name = os.path.basename(config_path).lower()
    if "offline" in config_name:
        return "offline"
    elif "online" in config_name:
        return "online"
    else:
        # 默认根据文件内容检测
        try:
            with open(config_path, "r") as f:
                content = f.read()
                if "dataset_mode: offline" in content or "offline_total_steps" in content:
                    return "offline"
                elif "dataset_mode: online" in content or "online_total_steps" in content:
                    return "online"
        except Exception:
            pass
    # 默认为 online
    return "online"


def build_run_cmd(
    python_bin: str,
    base_config: str,
    mode: str,
    label: str,
    flc: int,
    bps: int,
    milestone_steps,
    log_dir: str,
):
    """构建单次长训练的命令行（支持 online 和 offline 模式）"""
    milestone_steps = tuple(sorted(set(int(step) for step in milestone_steps)))
    max_steps = max(milestone_steps)
    event_interval_steps = _compute_event_interval_steps(milestone_steps)

    # 根据模式选择 run name 和参数
    mode_suffix = f"_{mode}" if mode == "offline" else ""
    run_name = f"{label}_flc{flc}_bps{bps}_steps{max_steps}{mode_suffix}"
    run_log_dir = os.path.join(log_dir, run_name)

    # 根据模式选择测试样本
    if mode == "offline":
        inference_samples = _resolve_inference_samples(
            OFFLINE_INFERENCE_SAMPLES,
            os.path.join(REPO_ROOT, "data", "input_data"),
        )
    else:
        inference_samples = _resolve_inference_samples(
            ONLINE_INFERENCE_SAMPLES,
            os.path.join(REPO_ROOT, "data", "online_eval_input_data"),
        )

    # 根据模式选择参数名称
    if mode == "offline":
        total_steps_param = "offline_total_steps"
        eval_interval_param = "offline_eval_interval_steps"
        save_interval_param = "offline_save_interval_steps"
        inference_interval_param = "offline_inference_test_interval_steps"
    else:
        total_steps_param = "online_total_steps"
        eval_interval_param = "online_eval_interval_steps"
        save_interval_param = "online_save_interval_steps"
        inference_interval_param = "online_inference_test_interval_steps"

    cmd = [
        python_bin,
        "train.py",
        "--config",
        base_config,
        "--first_layer_channels",
        str(flc),
        "--blocks_per_stage",
        str(bps),
        f"--{total_steps_param}",
        str(max_steps),
        f"--{eval_interval_param}",
        str(event_interval_steps),
        f"--{save_interval_param}",
        str(event_interval_steps),
        f"--{inference_interval_param}",
        str(event_interval_steps),
        "--log_dir",
        run_log_dir,
    ]

    # 添加 inference 相关参数（这些不在 config 文件中）
    inference_args = {
        "inference_num_cases": len(inference_samples),
        "inference_action_choice": "max",
        "sample_data_path": inference_samples,
    }

    for key, value in inference_args.items():
        _append_arg(cmd, key, value)

    return cmd, run_name, max_steps, event_interval_steps


def _parse_metric_kv(metric_str):
    metrics = {}
    for kv in metric_str.split(","):
        kv = kv.strip()
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def parse_train_output(output: str):
    """从训练输出中提取每个 milestone step 的指标"""
    metrics_by_step = {}
    current_step = None
    current_total_steps = None

    def ensure_step(step, total_steps):
        step = int(step)
        if step not in metrics_by_step:
            metrics_by_step[step] = {
                "reported_step": step,
                "reported_total_steps": int(total_steps),
            }
        return metrics_by_step[step]

    for raw_line in output.splitlines():
        # tqdm 用 \r 覆盖同一行，取最后一段才是真正的内容
        line = raw_line.rsplit("\r", 1)[-1].strip()
        if not line:
            continue

        # 某些日志会把 tqdm 进度条和指标打印粘在同一行，保留指标部分。
        for marker in ("Step ", "Inference case "):
            marker_idx = line.find(marker)
            if marker_idx > 0:
                line = line[marker_idx:]
                break

        match = STEP_PREFIX_RE.match(line)
        if match:
            current_step = int(match.group(1))
            current_total_steps = int(match.group(2))
            ensure_step(current_step, current_total_steps)

        match = TRAIN_LOSS_RE.match(line)
        if match:
            step, total_steps, loss = match.groups()
            row = ensure_step(step, total_steps)
            row["train_loss"] = float(loss)
            continue

        match = VAL_DIM_RE.match(line)
        if match:
            step, total_steps, width, height, loss = match.groups()
            row = ensure_step(step, total_steps)
            row[f"val_loss_{width}x{height}"] = float(loss)
            continue

        match = VAL_AGG_RE.match(line)
        if match:
            step, total_steps, loss = match.groups()
            row = ensure_step(step, total_steps)
            row["val_loss"] = float(loss)
            continue

        match = INFER_CASE_RE.match(line)
        if match:
            if current_step is None or current_total_steps is None:
                continue
            case_idx, file_name, metric_str = match.groups()
            row = ensure_step(current_step, current_total_steps)
            case_idx = int(case_idx)
            base_name = os.path.basename(file_name.strip())
            tag = _sanitize_name(os.path.splitext(base_name)[0])
            row[f"infer_case_{case_idx}_file"] = base_name
            row[f"infer_case_{case_idx}_step"] = row["reported_step"]
            for key, value in _parse_metric_kv(metric_str).items():
                row[f"infer_case_{case_idx}_{key}"] = value
                row[f"infer_{tag}_{key}"] = value
            continue

        match = INFER_SUMMARY_RE.match(line)
        if match:
            step, total_steps, metric_str = match.groups()
            row = ensure_step(step, total_steps)
            for key, value in _parse_metric_kv(metric_str).items():
                row[f"infer_{key}"] = value
            continue

    return metrics_by_step


def estimate_params(python_bin: str, flc: int, bps: int) -> int:
    """优先使用训练解释器估算参数量，避免当前解释器缺少 torch"""
    script = (
        "import sys; "
        "from models.unet import UNet; "
        "flc=int(sys.argv[1]); "
        "bps=int(sys.argv[2]); "
        "m=UNet(6, 5, first_layer_channels=flc, blocks_per_stage=bps); "
        "print(sum(p.numel() for p in m.parameters() if p.requires_grad))"
    )
    try:
        proc = subprocess.run(
            [python_bin, "-c", script, str(flc), str(bps)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return int(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return 0


def _build_rows_for_run(
    *,
    mode,
    label,
    flc,
    bps,
    n_params,
    source_run_name,
    requested_steps,
    metrics_by_step,
    wall_time_s,
    exit_code,
    batch_size,
):
    rows = []
    max_requested_step = max(requested_steps)
    requested_step_set = set(int(step) for step in requested_steps)

    for step in sorted(requested_step_set):
        parsed = dict(metrics_by_step.get(step, {}))
        mode_suffix = f"_{mode}" if mode == "offline" else ""
        row = {
            "run_name": f"{label}_flc{flc}_bps{bps}_steps{step}{mode_suffix}",
            "source_run_name": source_run_name,
            "label": label,
            "first_layer_channels": flc,
            "blocks_per_stage": bps,
            "n_params": n_params,
            "total_steps": step,
            "max_total_steps": max_requested_step,
            "data_tokens": step * batch_size,
            "source_wall_time_s": round(wall_time_s, 1),
            "exit_code": exit_code,
            "mode": mode,
            **parsed,
        }
        if step == max_requested_step:
            row["wall_time_s"] = round(wall_time_s, 1)
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Scaling law experiment runner (Unified version)")
    parser.add_argument("--config", required=True, help="Base config file (config.online.yaml or config.offline.yaml)")
    parser.add_argument("--python", default=sys.executable, help="Python binary")
    parser.add_argument("--log_dir", default="runs/scaling_law", help="Root log directory")
    parser.add_argument("--output_csv", default="scaling_law_results.csv", help="Output CSV path")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Only run these model labels (e.g. XS S M L)",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=None,
        help="Milestone steps collected from one long run (e.g. 10000 20000 40000)",
    )
    args = parser.parse_args()

    if not _python_has_torch(args.python):
        raise SystemExit(
            f"--python={args.python} 无法导入 torch。"
            "请传入真正训练环境的解释器，例如 /home/yimintan/anaconda3/envs/py38/bin/python"
        )

    # 检测模式
    mode = detect_mode(args.config)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{mode}_{timestamp}")

    models = MODEL_CONFIGS
    if args.models:
        models = [model for model in MODEL_CONFIGS if model[0] in args.models]
    steps = tuple(sorted(set(args.steps or STEP_CONFIGS)))
    if not steps:
        raise ValueError("At least one step milestone is required.")

    print(f"=== Scaling Law Experiment ({mode.upper()} mode) ===")
    print(f"Config: {args.config}")
    print(f"Mode: {mode}")
    print(f"Models: {[m[0] for m in models]}")
    print(f"Milestone steps: {list(steps)}")
    print(f"Long-run max step: {max(steps)}")
    print(f"Total train runs: {len(models)}")
    print(f"Output data points: {len(models) * len(steps)}")
    print(f"Log dir: {log_dir}")
    print(f"Output: {args.output_csv}")
    print()

    print("Model configs:")
    for label, flc, bps in models:
        n_params = estimate_params(args.python, flc, bps)
        print(f"  {label}: flc={flc}, bps={bps}, params={n_params/1e6:.1f}M")
    print()

    if args.dry_run:
        print("=== Dry Run (commands only) ===")
        for label, flc, bps in models:
            cmd, run_name, _, _ = build_run_cmd(
                args.python, args.config, mode, label, flc, bps, steps, log_dir
            )
            print(f"[{run_name}] {' '.join(cmd)}")
        return

    os.makedirs(log_dir, exist_ok=True)
    results = []

    for i, (label, flc, bps) in enumerate(models):
        cmd, source_run_name, max_steps, event_interval_steps = build_run_cmd(
            args.python,
            args.config,
            mode,
            label,
            flc,
            bps,
            steps,
            log_dir,
        )
        n_params = estimate_params(args.python, flc, bps)

        print(
            f"\n[{i+1}/{len(models)}] {source_run_name} "
            f"(params={n_params/1e6:.1f}M, max_steps={max_steps}, interval={event_interval_steps}, points={len(steps)})"
        )
        print(f"  cmd: {' '.join(cmd)}")

        t0 = time.time()
        try:
            # 直接继承终端，让 tqdm 进度条正常显示
            # 同时用 tee 把输出写到日志文件用于解析
            run_log_path = os.path.join(log_dir, f"{source_run_name}.log")
            with open(run_log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,  # 无缓冲
                )

                # 逐块读取，保留 \r 让终端正确渲染进度条
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    log_file.write(chunk.decode("utf-8", errors="replace"))

            proc.wait(timeout=36000)
            wall_time = time.time() - t0

            with open(run_log_path, "r", encoding="utf-8") as f:
                output = f.read()

            metrics_by_step = parse_train_output(output)
            rows = _build_rows_for_run(
                mode=mode,
                label=label,
                flc=flc,
                bps=bps,
                n_params=n_params,
                source_run_name=source_run_name,
                requested_steps=steps,
                metrics_by_step=metrics_by_step,
                wall_time_s=wall_time,
                exit_code=proc.returncode,
                batch_size=64,
            )
            results.extend(rows)

            parsed_steps = sorted(metrics_by_step.keys())
            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
            print(f"\n  {status} in {wall_time:.0f}s | parsed milestones={parsed_steps}")

            if proc.returncode != 0:
                err_path = os.path.join(log_dir, f"{source_run_name}.stderr.log")
                with open(err_path, "w", encoding="utf-8") as file:
                    file.write(output)
                print(f"  error log: {err_path}")

        except subprocess.TimeoutExpired:
            wall_time = time.time() - t0
            proc.kill()
            proc.wait()
            try:
                with open(run_log_path, "r", encoding="utf-8") as f:
                    output = f.read()
            except Exception:
                output = ""

            metrics_by_step = parse_train_output(output)
            rows = _build_rows_for_run(
                mode=mode,
                label=label,
                flc=flc,
                bps=bps,
                n_params=n_params,
                source_run_name=source_run_name,
                requested_steps=steps,
                metrics_by_step=metrics_by_step,
                wall_time_s=wall_time,
                exit_code=-1,
                batch_size=64,
            )
            results.extend(rows)
            print(f"\n  TIMEOUT after {wall_time:.0f}s | parsed milestones={sorted(metrics_by_step.keys())}")

        _write_csv(args.output_csv, results)

    print(f"\n=== Done: {len(models)} train runs, {len(results)} CSV rows ===")
    print(f"Results saved to: {args.output_csv}")


def _write_csv(path: str, results: list):
    if not results:
        return
    all_keys = list(dict.fromkeys(key for row in results for key in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
