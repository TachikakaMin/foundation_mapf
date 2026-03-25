#!/usr/bin/env python
"""Plot comprehensive scaling law results from a CSV, leaving gaps for missing data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_scaling_law import (
    COLORS,
    fit_scaling_law,
    load_rows,
    ordered_labels,
    param_map,
    safe_float,
    safe_int,
    scaling_law_fn,
    series_by_label,
    series_by_step,
    step_grid,
)


plt.rcParams.update({"font.size": 11})

CASE_GROUPS = [
    ("32x32, 128 agents", [0, 1, 2, 3], "#1f77b4"),
    ("64x64, 256 agents", [4, 5, 6, 7], "#ff7f0e"),
    ("64x64, 1024 agents", [10, 11], "#d62728"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="scaling_law_results.csv", help="input CSV path")
    parser.add_argument("--output", default="scaling_law_full_plot.png", help="output PNG path")
    return parser.parse_args()


def group_metric(row, case_indices, metric_name):
    values = []
    for case_idx in case_indices:
        value = safe_float(row.get(f"infer_case_{case_idx}_{metric_name}"))
        if np.isfinite(value):
            values.append(value)
    if len(values) != len(case_indices):
        return np.nan
    return float(np.mean(values))


def group_series(rows, label, step_vals, case_indices, metric_name):
    lookup = {}
    for row in rows:
        if row.get("label") != label:
            continue
        step = safe_int(row.get("total_steps"))
        if step is None:
            continue
        lookup[step] = group_metric(row, case_indices, metric_name)
    return np.array([lookup.get(step, np.nan) for step in step_vals], dtype=float)


def main():
    args = parse_args()
    all_rows, valid_rows = load_rows(args.csv)
    labels = ordered_labels(all_rows)
    step_vals = step_grid(all_rows)
    params = param_map(all_rows, labels)
    fit_result = fit_scaling_law(valid_rows)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("RAILGUN Scaling Law - Current Snapshot", fontsize=16, fontweight="bold", y=0.995)

    ax = axes[0, 0]
    for label in labels:
        y_vals = series_by_label(valid_rows, label, step_vals, "train_loss")
        if not np.isfinite(y_vals).any():
            continue
        ax.plot(
            step_vals,
            y_vals,
            "o-",
            color=COLORS.get(label),
            linewidth=2,
            markersize=5,
            label=f"{label} ({params.get(label, np.nan) / 1e6:.0f}M)" if label in params else label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Train Loss")
    ax.set_title("(a) Train Loss vs Steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for label in labels:
        y_vals = series_by_label(all_rows, label, step_vals, "val_loss")
        if not np.isfinite(y_vals).any():
            continue
        ax.plot(
            step_vals,
            y_vals,
            "o-",
            color=COLORS.get(label),
            linewidth=2,
            markersize=5,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(b) Validation Loss vs Steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    x_labels = [label for label in labels if label in params]
    x_params = np.array([params[label] for label in x_labels], dtype=float)
    step_colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(step_vals), 2)))
    for idx, step in enumerate(step_vals):
        y_vals = series_by_step(valid_rows, step, x_labels, "train_loss")
        if not np.isfinite(y_vals).any():
            continue
        ax.plot(
            x_params,
            y_vals,
            "o-",
            color=step_colors[idx],
            linewidth=2,
            markersize=5,
            label=f"{step // 1000}k",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)")
    ax.set_ylabel("Train Loss")
    ax.set_title("(c) Loss vs Model Size")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if fit_result is not None:
        popt, pred, actual, r2 = fit_result
        for label in labels:
            label_rows = [row for row in valid_rows if row.get("label") == label]
            if not label_rows:
                continue
            label_actual = np.array([safe_float(row["train_loss"]) for row in label_rows], dtype=float)
            label_pred = scaling_law_fn(
                (
                    np.array([safe_float(row["n_params"]) for row in label_rows], dtype=float),
                    np.array([safe_int(row["total_steps"]) * 64 for row in label_rows], dtype=float),
                ),
                *popt,
            )
            ax.scatter(label_pred, label_actual, color=COLORS.get(label), s=50, label=label)
        lims = [min(actual.min(), pred.min()) * 0.95, max(actual.max(), pred.max()) * 1.05]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Predicted Loss")
        ax.set_ylabel("Actual Loss")
        ax.set_title(f"(d) Scaling Law Fit (R^2={r2:.4f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Fit failed", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(d) Scaling Law Fit")

    ax = axes[1, 1]
    for label in labels:
        y_vals = series_by_label(all_rows, label, step_vals, "infer_isr")
        if not np.isfinite(y_vals).any():
            continue
        ax.plot(
            step_vals,
            y_vals,
            "o-",
            color=COLORS.get(label),
            linewidth=2,
            markersize=5,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("ISR")
    ax.set_title("(e) Inference ISR vs Steps (avg over available cases)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for label in labels:
        y_vals = series_by_label(all_rows, label, step_vals, "infer_final_distance")
        if not np.isfinite(y_vals).any():
            continue
        ax.plot(
            step_vals,
            y_vals,
            "o-",
            color=COLORS.get(label),
            linewidth=2,
            markersize=5,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Final Distance")
    ax.set_title("(f) Final Distance vs Steps (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for group_idx, (axis, (title, case_indices, color)) in enumerate(zip(axes[2], CASE_GROUPS)):
        for label in labels:
            y_vals = group_series(all_rows, label, step_vals, case_indices, "isr")
            if not np.isfinite(y_vals).any():
                continue
            axis.plot(
                step_vals,
                y_vals,
                "o-",
                color=COLORS.get(label),
                linewidth=2,
                markersize=5,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Training Steps")
        axis.set_ylabel("ISR")
        axis.set_title(f"({chr(ord('g') + group_idx)}) {title}")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    if fit_result is not None:
        popt = fit_result[0]
        r2 = fit_result[3]
        E, A, alpha, B, beta = popt
        equation = (
            f"L(N,D) = {E:.4f} + {A:.2e}/N^{{{alpha:.3f}}} + "
            f"{B:.2e}/D^{{{beta:.3f}}}   (R^2={r2:.4f})"
        )
        fig.text(0.5, 0.002, equation, ha="center", fontsize=11, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
