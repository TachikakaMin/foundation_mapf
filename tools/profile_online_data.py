import argparse
import glob
import os
import sys
import statistics
import time

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MAPF_online_dataset import MAPFOnlineBufferLoader


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _get_cfg_value(cfg, key, default=None):
    value = cfg.get(key, default)
    return value if value is not None else default


def _format_seconds(values):
    if not values:
        return "n/a"
    return ", ".join(f"{v:.4f}s" for v in values)


def _summarize(times):
    times_sorted = sorted(times)
    mean_v = float(statistics.mean(times_sorted))
    p50 = float(np.percentile(times_sorted, 50))
    p90 = float(np.percentile(times_sorted, 90))
    p99 = float(np.percentile(times_sorted, 99)) if len(times_sorted) >= 2 else times_sorted[-1]
    return {
        "count": len(times_sorted),
        "mean": mean_v,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "max": max(times_sorted),
    }


def _print_summary(label, summary):
    print(f"{label} count={summary['count']}")
    print(
        "  mean={:.4f}s  p50={:.4f}s  p90={:.4f}s  p99={:.4f}s  max={:.4f}s".format(
            summary["mean"],
            summary["p50"],
            summary["p90"],
            summary["p99"],
            summary["max"],
        )
    )


def _profile_scenarios(dataset, count, seed):
    if count <= 0:
        return None

    rng = np.random.default_rng(seed)
    pair_counters = np.zeros(len(dataset._pairs), dtype=np.int64)
    generation_times = []
    step_counts = []
    success_count = 0
    attempt_count = 0

    for idx in range(count):
        t0 = time.perf_counter()
        attempts = dataset._prepare_attempts(
            rng=rng,
            pair_counters=pair_counters,
            worker_id=0,
            num_workers=1,
        )
        attempt_count += len(attempts)
        result = dataset._generate_scenario_from_attempts(attempts)
        dt = time.perf_counter() - t0
        generation_times.append(dt)

        step_count = -1
        if result is not None:
            success_count += 1
            step_count = int(result[2].shape[0])
            step_counts.append(step_count)
        print(
            f"scenario {idx + 1}/{count}: gen={dt:.4f}s success={result is not None} "
            f"steps={step_count}"
        )

    return {
        "success_count": success_count,
        "attempt_count": attempt_count,
        "generation_summary": _summarize(generation_times),
        "step_summary": _summarize(step_counts) if step_counts else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile online data loading latency.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.online.yaml",
        help="Path to online config yaml.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=20,
        help="Number of batches to time.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup batches to skip before timing.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override num_workers for profiling.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch_size for profiling.",
    )
    parser.add_argument(
        "--scenario_samples",
        type=int,
        default=0,
        help="Number of raw scenario generations to time; 0 disables scenario profiling.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train_map_path = _get_cfg_value(cfg, "train_map_path", "data/map_files")
    map_files = sorted(glob.glob(os.path.join(train_map_path, "**/*.map"), recursive=True))
    if not map_files:
        raise RuntimeError(f"No map files found under: {train_map_path}")

    feature_dim = int(_get_cfg_value(cfg, "feature_dim", 6))
    feature_type = _get_cfg_value(cfg, "feature_type", "gradient")
    seed = int(_get_cfg_value(cfg, "seed", 1919180))
    time_limit_sec = int(_get_cfg_value(cfg, "online_time_limit_sec", 2))
    retry_limit = int(_get_cfg_value(cfg, "online_retry_limit", 2))
    buffer_size = int(_get_cfg_value(cfg, "online_buffer_size", 512))
    buffer_workers = int(_get_cfg_value(cfg, "num_workers", 4))
    buffer_timeout_sec = float(_get_cfg_value(cfg, "online_buffer_timeout_sec", 1.0))

    batch_size = int(args.batch_size if args.batch_size is not None else _get_cfg_value(cfg, "batch_size", 64))
    if args.num_workers is not None:
        buffer_workers = int(args.num_workers)

    loader = MAPFOnlineBufferLoader(
        map_files,
        feature_dim,
        feature_type,
        batch_size=batch_size,
        seed=seed,
        time_limit_sec=time_limit_sec,
        retry_limit=retry_limit,
        buffer_size=buffer_size,
        buffer_workers=buffer_workers,
        buffer_timeout_sec=buffer_timeout_sec,
    )
    try:
        it = iter(loader)
        warmup = max(0, args.warmup)
        for _ in range(warmup):
            _ = next(it)

        times = []
        for idx in range(args.batches):
            t0 = time.perf_counter()
            _ = next(it)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            print(f"batch {idx + 1}/{args.batches}: fetch={times[-1]:.4f}s")

        summary = _summarize(times)
        print("=== Online Step Buffer Profile ===")
        print(f"config={args.config}")
        print(f"map_files={len(map_files)}")
        print(
            f"batch_size={batch_size} buffer_size={buffer_size} "
            f"buffer_workers={buffer_workers} buffer_timeout_sec={buffer_timeout_sec}"
        )
        _print_summary("fetch_time", summary)

        scenario_profile = _profile_scenarios(loader._dataset, args.scenario_samples, seed + 97)
        if scenario_profile is not None:
            print("=== Scenario Generation Profile ===")
            print(
                f"successes={scenario_profile['success_count']} "
                f"attempts_total={scenario_profile['attempt_count']}"
            )
            _print_summary("scenario_gen_time", scenario_profile["generation_summary"])
            if scenario_profile["step_summary"] is not None:
                step_summary = scenario_profile["step_summary"]
                print(f"trajectory_steps count={step_summary['count']}")
                print(
                    "  mean={:.2f}  p50={:.2f}  p90={:.2f}  p99={:.2f}  max={:.2f}".format(
                        step_summary["mean"],
                        step_summary["p50"],
                        step_summary["p90"],
                        step_summary["p99"],
                        step_summary["max"],
                    )
                )
    finally:
        loader.close()


if __name__ == "__main__":
    main()
