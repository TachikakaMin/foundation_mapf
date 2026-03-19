#!/usr/bin/env python
"""
RAILGUN Scaling Law 实验脚本。
Scaling law experiment runner.

用法 / Usage:
    python scaling_law.py --config config.online.yaml [--dry_run]

实验设计 / Experiment design:
    - N 轴（模型大小）: 4 个配置，通过 first_layer_channels × blocks_per_stage 控制
    - D 轴（数据量）: 4-5 个 online_total_steps 值
    - 每组实验记录: final train loss, val loss, inference metrics (CSR, ISR, makespan)
    - 拟合目标: L(N, D) = E + A/N^α + B/D^β
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime


# ── 实验 Grid 定义 ──────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    # (label, first_layer_channels, blocks_per_stage)
    ("XS", 32, 1),
    ("S",  64, 1),
    ("M",  64, 2),
    ("L", 128, 1),
]

STEP_CONFIGS = [500, 1000, 2000, 5000]

# ── 固定参数 ────────────────────────────────────────────────────────────────

FIXED_ARGS = {
    "seed": 1919180,
    "batch_size": 64,
    "num_workers": 20,
    "inference_num_cases": 1,
    "inference_action_choice": "max",
}


def build_run_cmd(
    python_bin: str,
    base_config: str,
    label: str,
    flc: int,
    bps: int,
    total_steps: int,
    log_dir: str,
):
    """构建单次训练的命令行。"""
    run_name = f"{label}_flc{flc}_bps{bps}_steps{total_steps}"
    run_log_dir = os.path.join(log_dir, run_name)

    cmd = [
        python_bin, "train.py",
        "--config", base_config,
        "--first_layer_channels", str(flc),
        "--blocks_per_stage", str(bps),
        "--online_total_steps", str(total_steps),
        # eval 和 save 只在最后一步做
        "--online_eval_interval_steps", str(total_steps),
        "--online_save_interval_steps", str(total_steps),
        "--online_inference_test_interval_steps", str(total_steps),
        "--log_dir", run_log_dir,
    ]
    for key, value in FIXED_ARGS.items():
        cmd.extend([f"--{key}", str(value)])

    return cmd, run_name


def parse_train_output(output: str):
    """从训练输出中提取最终指标。"""
    results = {}

    # 提取最终 training loss: "Step N/N, Training mean Loss: X.XXX"
    for line in output.split("\n"):
        if "Training mean Loss:" in line:
            try:
                results["train_loss"] = float(line.split("Training mean Loss:")[-1].strip())
            except ValueError:
                pass

        # 提取 val loss: "Val loss (32x32): X.XXXX"
        if "Val loss" in line and ":" in line:
            try:
                val_str = line.split(":")[-1].strip()
                results["val_loss"] = float(val_str)
            except ValueError:
                pass

        # 提取 inference summary
        if "Inference summary:" in line:
            summary_part = line.split("Inference summary:")[-1].strip()
            for kv in summary_part.split(","):
                kv = kv.strip()
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        results[f"infer_{k.strip()}"] = float(v.strip())
                    except ValueError:
                        pass

    return results


def estimate_params(flc: int, bps: int) -> int:
    """粗略估算参数量（不需要实际构建模型）。"""
    try:
        import torch
        sys.path.insert(0, ".")
        from models.unet import UNet
        m = UNet(6, 5, first_layer_channels=flc, blocks_per_stage=bps)
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Scaling law experiment runner")
    parser.add_argument("--config", default="config.online.yaml", help="Base config file")
    parser.add_argument("--python", default=sys.executable, help="Python binary")
    parser.add_argument("--log_dir", default="runs/scaling_law", help="Root log directory")
    parser.add_argument("--output_csv", default="scaling_law_results.csv", help="Output CSV path")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only run these model labels (e.g. XS S M L)")
    parser.add_argument("--steps", nargs="*", type=int, default=None,
                        help="Only run these step counts (e.g. 500 1000 2000)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)

    models = MODEL_CONFIGS
    if args.models:
        models = [m for m in MODEL_CONFIGS if m[0] in args.models]
    steps = args.steps or STEP_CONFIGS

    experiments = list(itertools.product(models, steps))
    print(f"=== Scaling Law Experiment ===")
    print(f"Models: {[m[0] for m in models]}")
    print(f"Steps: {steps}")
    print(f"Total runs: {len(experiments)}")
    print(f"Log dir: {log_dir}")
    print(f"Output: {args.output_csv}")
    print()

    # 预估参数量
    print("Model configs:")
    for label, flc, bps in models:
        n_params = estimate_params(flc, bps)
        print(f"  {label}: flc={flc}, bps={bps}, params={n_params/1e6:.1f}M")
    print()

    if args.dry_run:
        print("=== Dry Run (commands only) ===")
        for (label, flc, bps), total_steps in experiments:
            cmd, run_name = build_run_cmd(args.python, args.config, label, flc, bps, total_steps, log_dir)
            print(f"[{run_name}] {' '.join(cmd)}")
        return

    # 运行实验
    os.makedirs(log_dir, exist_ok=True)
    results = []

    for i, ((label, flc, bps), total_steps) in enumerate(experiments):
        cmd, run_name = build_run_cmd(args.python, args.config, label, flc, bps, total_steps, log_dir)
        n_params = estimate_params(flc, bps)
        data_tokens = total_steps * FIXED_ARGS["batch_size"]

        print(f"\n[{i+1}/{len(experiments)}] {run_name} (params={n_params/1e6:.1f}M, D={data_tokens})")
        print(f"  cmd: {' '.join(cmd)}")

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2h max per run
            )
            wall_time = time.time() - t0
            output = proc.stdout + "\n" + proc.stderr
            metrics = parse_train_output(output)

            row = {
                "run_name": run_name,
                "label": label,
                "first_layer_channels": flc,
                "blocks_per_stage": bps,
                "n_params": n_params,
                "total_steps": total_steps,
                "data_tokens": data_tokens,
                "wall_time_s": round(wall_time, 1),
                "exit_code": proc.returncode,
                **metrics,
            }
            results.append(row)

            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
            loss_str = f"train_loss={metrics.get('train_loss', '?')}"
            print(f"  {status} in {wall_time:.0f}s | {loss_str}")

            if proc.returncode != 0:
                # 保存错误日志
                err_path = os.path.join(log_dir, f"{run_name}.stderr.log")
                with open(err_path, "w") as f:
                    f.write(output)
                print(f"  error log: {err_path}")

        except subprocess.TimeoutExpired:
            wall_time = time.time() - t0
            print(f"  TIMEOUT after {wall_time:.0f}s")
            results.append({
                "run_name": run_name, "label": label,
                "first_layer_channels": flc, "blocks_per_stage": bps,
                "n_params": n_params, "total_steps": total_steps,
                "data_tokens": data_tokens, "wall_time_s": round(wall_time, 1),
                "exit_code": -1,
            })

        # 每跑完一组就写 CSV（防止中途挂掉丢数据）
        _write_csv(args.output_csv, results)

    print(f"\n=== Done: {len(results)} runs ===")
    print(f"Results saved to: {args.output_csv}")


def _write_csv(path: str, results: list):
    if not results:
        return
    all_keys = list(dict.fromkeys(k for r in results for k in r.keys()))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
