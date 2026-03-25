#!/usr/bin/env python
"""Plot scaling law curves from a CSV, leaving gaps for missing data."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


LABEL_ORDER = ["XS", "S", "M", "L", "XL"]
COLORS = {
    "XS": "#1f77b4",
    "S": "#ff7f0e",
    "M": "#2ca02c",
    "L": "#d62728",
    "XL": "#9467bd",
}
WT_PER_STEP = {"XS": 0.0253, "S": 0.0370, "M": 0.0597, "L": 0.0822, "XL": 0.1045}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="scaling_law_results.csv", help="input CSV path")
    parser.add_argument("--output", default="scaling_law_plot.png", help="output PNG path")
    return parser.parse_args()


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_rows(csv_path):
    all_rows = []
    valid_rows = []
    with open(csv_path, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            all_rows.append(row)
            if np.isfinite(safe_float(row.get("train_loss"))) and safe_int(row.get("exit_code")) == 0:
                valid_rows.append(row)
    return all_rows, valid_rows


def ordered_labels(rows):
    present = {row["label"] for row in rows if row.get("label")}
    ordered = [label for label in LABEL_ORDER if label in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def step_grid(rows):
    return sorted({safe_int(row.get("total_steps")) for row in rows if safe_int(row.get("total_steps")) is not None})


def param_map(rows, labels):
    result = {}
    for label in labels:
        for row in rows:
            n_params = safe_float(row.get("n_params"))
            if row.get("label") == label and np.isfinite(n_params) and n_params > 0:
                result[label] = safe_float(row["n_params"])
                break
    return result


def series_by_label(rows, label, step_vals, metric_key):
    lookup = {}
    for row in rows:
        if row.get("label") != label:
            continue
        step = safe_int(row.get("total_steps"))
        if step is None:
            continue
        lookup[step] = safe_float(row.get(metric_key))
    return np.array([lookup.get(step, np.nan) for step in step_vals], dtype=float)


def series_by_step(rows, step, labels, metric_key):
    lookup = {}
    for row in rows:
        row_step = safe_int(row.get("total_steps"))
        if row_step != step:
            continue
        lookup[row.get("label")] = safe_float(row.get(metric_key))
    return np.array([lookup.get(label, np.nan) for label in labels], dtype=float)


def latest_step_by_label(rows, labels):
    result = {}
    for label in labels:
        vals = [
            safe_int(row.get("total_steps"))
            for row in rows
            if row.get("label") == label and np.isfinite(safe_float(row.get("train_loss")))
        ]
        result[label] = max(vals) if vals else None
    return result


def scaling_law_fn(inputs, E, A, alpha, B, beta):
    n_params, data_tokens = inputs
    return E + A / np.power(n_params, alpha) + B / np.power(data_tokens, beta)


def fit_scaling_law(rows):
    fit_rows = [
        row
        for row in rows
        if safe_float(row.get("n_params")) > 0 and safe_int(row.get("total_steps")) not in (None, 0)
    ]
    if len(fit_rows) < 5:
        return None
    n_arr = np.array([safe_float(row["n_params"]) for row in fit_rows], dtype=float)
    d_arr = np.array([safe_int(row["total_steps"]) * 64 for row in fit_rows], dtype=float)
    l_arr = np.array([safe_float(row["train_loss"]) for row in fit_rows], dtype=float)
    try:
        popt, _ = curve_fit(
            scaling_law_fn,
            (n_arr, d_arr),
            l_arr,
            p0=[0.1, 1e3, 0.3, 1e2, 0.3],
            bounds=([0, 0, 0.01, 0, 0.01], [2, 1e10, 2, 1e10, 2]),
            maxfev=50000,
        )
    except Exception as exc:
        print(f"Fit failed: {exc}")
        return None

    pred = scaling_law_fn((n_arr, d_arr), *popt)
    residuals = l_arr - pred
    r2 = 1 - np.sum(residuals ** 2) / np.sum((l_arr - np.mean(l_arr)) ** 2)
    return popt, pred, l_arr, r2


def main():
    args = parse_args()
    all_rows, valid_rows = load_rows(args.csv)
    labels = ordered_labels(all_rows)
    step_vals = step_grid(all_rows)
    params = param_map(all_rows, labels)
    fit_result = fit_scaling_law(valid_rows)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    for label in labels:
        y_vals = series_by_label(valid_rows, label, step_vals, "train_loss")
        if not np.isfinite(y_vals).any():
            continue
        ax.plot(
            step_vals,
            y_vals,
            "o-",
            color=COLORS.get(label, None),
            label=f"{label} ({params.get(label, np.nan) / 1e6:.0f}M)" if label in params else label,
            linewidth=2,
            markersize=6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Train Loss")
    ax.set_title("(a) Loss vs Steps")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    x_params = np.array([params[label] for label in labels if label in params], dtype=float)
    x_labels = [label for label in labels if label in params]
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
            label=f"{step // 1000}k steps",
            linewidth=2,
            markersize=6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)")
    ax.set_ylabel("Train Loss")
    ax.set_title("(b) Loss vs Model Size")
    ax.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
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
            ax.scatter(label_pred, label_actual, color=COLORS.get(label, None), s=60, zorder=3, label=label)
        lims = [min(actual.min(), pred.min()) * 0.95, max(actual.max(), pred.max()) * 1.05]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Predicted Loss")
        ax.set_ylabel("Actual Loss")
        ax.set_title(f"(c) Fit Quality (R²={r2:.4f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Fit failed", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 1]
    if fit_result is not None:
        popt, _, _, _ = fit_result
        budget_hours = [0.5, 1, 2, 4, 8]
        budget_labels = [label for label in labels if label in params and label in WT_PER_STEP]
        budget_params = np.array([params[label] for label in budget_labels], dtype=float)
        x_pos = np.arange(len(budget_labels), dtype=float)
        for budget_h in budget_hours:
            budget_s = budget_h * 3600
            losses = []
            for label in budget_labels:
                max_steps = int(budget_s / (WT_PER_STEP[label] + 0.003))
                data_tokens = max_steps * 64
                losses.append(scaling_law_fn((np.array([params[label]]), np.array([data_tokens])), *popt)[0])
            ax.plot(
                x_pos,
                losses,
                "o-",
                linewidth=2,
                markersize=6,
                label=f"{budget_h}h budget",
            )
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Predicted Loss")
        ax.set_title("(d) Optimal Model Size per Wall-Time Budget")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(budget_labels)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Fit failed", ha="center", va="center", transform=ax.transAxes)

    if fit_result is not None:
        popt, _, _, r2 = fit_result
        E, A, alpha, B, beta = popt
        equation = (
            f"L(N,D) = {E:.4f} + {A:.2e}/N^{{{alpha:.3f}}} + {B:.2e}/D^{{{beta:.3f}}}"
            f"    (R²={r2:.4f})"
        )
        fig.text(0.5, -0.01, equation, ha="center", fontsize=12, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    print(f"\nData points: {len(valid_rows)} (valid) / {len(all_rows)} (total)")
    max_steps = latest_step_by_label(valid_rows, labels)
    print("Latest available step per model:")
    for label in labels:
        print(f"  {label}: {max_steps.get(label)}")
    if fit_result is not None:
        E, A, alpha, B, beta = fit_result[0]
        r2 = fit_result[3]
        print(f"\nFit: L(N,D) = {E:.4f} + {A:.2e}/N^{alpha:.3f} + {B:.2e}/D^{beta:.3f}")
        print(f"R² = {r2:.4f}")


if __name__ == "__main__":
    main()
