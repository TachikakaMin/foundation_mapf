#!/usr/bin/env python
"""Plot a compact summary of the current scaling law snapshot."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_scaling_law import (
    COLORS,
    load_rows,
    ordered_labels,
    param_map,
    safe_float,
    safe_int,
    series_by_step,
    step_grid,
)


plt.rcParams.update({"font.size": 10})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="scaling_law_results.csv", help="input CSV path")
    parser.add_argument("--output", default="scaling_law_summary.png", help="output PNG path")
    return parser.parse_args()


def latest_metric_row(rows, label, metric_key):
    candidates = []
    for row in rows:
        if row.get("label") != label:
            continue
        value = safe_float(row.get(metric_key))
        step = safe_int(row.get("total_steps"))
        if np.isfinite(value) and step is not None:
            candidates.append((step, row))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def group_metric(row, case_indices, metric_name):
    values = []
    for case_idx in case_indices:
        value = safe_float(row.get(f"infer_case_{case_idx}_{metric_name}"))
        if np.isfinite(value):
            values.append(value)
    if len(values) != len(case_indices):
        return np.nan
    return float(np.mean(values))


def latest_group_metric_row(rows, label, case_indices, metric_name):
    candidates = []
    for row in rows:
        if row.get("label") != label:
            continue
        step = safe_int(row.get("total_steps"))
        value = group_metric(row, case_indices, metric_name)
        if step is not None and np.isfinite(value):
            candidates.append((step, row))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def select_steps(step_vals):
    targets = [5000, 10000, 20000, 30000, 50000, 100000]
    selected = [step for step in targets if step in step_vals]
    return selected if selected else step_vals


def annotate_steps(ax, xs, ys, steps):
    shown_steps = [step for step in steps if step is not None]
    if len(set(shown_steps)) <= 1:
        return
    for x_val, y_val, step in zip(xs, ys, steps):
        if not np.isfinite(y_val) or step is None:
            continue
        ax.text(x_val, y_val, f" {step // 1000}k", fontsize=8, va="bottom", ha="left")


def main():
    args = parse_args()
    all_rows, valid_rows = load_rows(args.csv)
    labels = ordered_labels(all_rows)
    params = param_map(all_rows, labels)
    step_vals = step_grid(all_rows)
    chosen_steps = select_steps(step_vals)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("RAILGUN Scaling Law - Current Summary", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    x_labels = [label for label in labels if label in params]
    x_params = np.array([params[label] for label in x_labels], dtype=float)
    step_colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(chosen_steps), 2)))
    for idx, step in enumerate(chosen_steps):
        y_vals = series_by_step(valid_rows, step, x_labels, "train_loss")
        if not np.isfinite(y_vals).any():
            continue
        ax.plot(
            x_params,
            y_vals,
            "o-",
            color=step_colors[idx],
            linewidth=2.5,
            markersize=7,
            label=f"{step // 1000}k steps",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters (N)")
    ax.set_ylabel("Train Loss")
    ax.set_title("(a) Loss vs Model Size")
    ax.legend(fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=True)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    x_pos = np.arange(len(labels))
    width = 0.36
    train_rows = [latest_metric_row(all_rows, label, "train_loss") for label in labels]
    val_rows = [latest_metric_row(all_rows, label, "val_loss") for label in labels]
    train_vals = np.array([safe_float(row.get("train_loss")) if row else np.nan for row in train_rows], dtype=float)
    val_vals = np.array([safe_float(row.get("val_loss")) if row else np.nan for row in val_rows], dtype=float)
    ax.bar(x_pos - width / 2, train_vals, width=width, color="#4c78a8", alpha=0.85, label="Train")
    ax.bar(x_pos + width / 2, val_vals, width=width, color="#f58518", alpha=0.85, label="Validation")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("Loss")
    ax.set_title("(b) Latest Available Train / Val Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    annotate_steps(
        ax,
        x_pos - width / 2,
        train_vals,
        [safe_int(row.get("total_steps")) if row else None for row in train_rows],
    )
    annotate_steps(
        ax,
        x_pos + width / 2,
        val_vals,
        [safe_int(row.get("total_steps")) if row else None for row in val_rows],
    )

    ax = axes[1, 0]
    infer_rows = [latest_metric_row(all_rows, label, "infer_isr") for label in labels]
    infer_vals = np.array([safe_float(row.get("infer_isr")) if row else np.nan for row in infer_rows], dtype=float)
    ax.bar(
        x_pos,
        infer_vals,
        color=[COLORS.get(label, "#999999") for label in labels],
        alpha=0.85,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("ISR")
    ax.set_title("(c) Latest Available Inference ISR")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(0.6, np.nanmax(infer_vals) * 1.2 if np.isfinite(infer_vals).any() else 0.6))
    annotate_steps(
        ax,
        x_pos,
        infer_vals,
        [safe_int(row.get("total_steps")) if row else None for row in infer_rows],
    )

    ax = axes[1, 1]
    case_indices = [10, 11]
    high_density_rows = [latest_group_metric_row(all_rows, label, case_indices, "isr") for label in labels]
    high_density_vals = np.array(
        [group_metric(row, case_indices, "isr") if row else np.nan for row in high_density_rows],
        dtype=float,
    )
    ax.bar(
        x_pos,
        high_density_vals,
        color=[COLORS.get(label, "#999999") for label in labels],
        alpha=0.85,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("ISR")
    ax.set_title("(d) Latest 64x64 / 1024-Agent ISR")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(0.35, np.nanmax(high_density_vals) * 1.25 if np.isfinite(high_density_vals).any() else 0.35))
    annotate_steps(
        ax,
        x_pos,
        high_density_vals,
        [safe_int(row.get("total_steps")) if row else None for row in high_density_rows],
    )

    fig.text(0.5, 0.02, "Latest bars use the highest available checkpoint for each model.", ha="center", fontsize=11, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
