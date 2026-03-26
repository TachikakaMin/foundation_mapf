#!/usr/bin/env python
"""Plot scaling-law figures in a DeepFleet-like style."""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plot_scaling_law import (
    load_rows,
    ordered_labels,
    param_map,
    safe_float,
    safe_int,
)


plt.rcParams.update({"font.size": 11})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="scaling_law_results.csv", help="input CSV path")
    parser.add_argument(
        "--output_curves",
        default="scaling_law_deepfleet_curves.png",
        help="output PNG for loss/data/FLOPs/isoFLOP curves",
    )
    parser.add_argument(
        "--output_optimal",
        default="scaling_law_deepfleet_optimal.png",
        help="output PNG for optimal data/model scaling",
    )
    parser.add_argument(
        "--flop_coeff",
        type=float,
        default=6.0,
        help="training FLOP proxy coefficient in coeff * N * D",
    )
    parser.add_argument(
        "--num_isoflop_curves",
        type=int,
        default=5,
        help="maximum number of isoFLOP target curves to draw",
    )
    return parser.parse_args()


def enrich_rows(rows, flop_coeff):
    enriched = []
    for row in rows:
        n_params = safe_float(row.get("n_params"))
        step = safe_int(row.get("total_steps"))
        train_loss = safe_float(row.get("train_loss"))
        if not (np.isfinite(n_params) and n_params > 0 and step and np.isfinite(train_loss)):
            continue
        data_tokens = safe_float(row.get("data_tokens"))
        if not np.isfinite(data_tokens) or data_tokens <= 0:
            data_tokens = step * 64
        enriched.append(
            {
                **row,
                "_n_params": n_params,
                "_step": step,
                "_train_loss": train_loss,
                "_data_tokens": data_tokens,
                "_train_flops": flop_coeff * n_params * data_tokens,
            }
        )
    return enriched


def series_lookup(rows, label, key):
    points = sorted(
        ((row[key], row["_train_loss"]) for row in rows if row.get("label") == label),
        key=lambda item: item[0],
    )
    if not points:
        return np.array([]), np.array([])
    xs, ys = zip(*points)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def shared_budget_range(rows, labels):
    mins = []
    maxs = []
    for label in labels:
        label_flops = [row["_train_flops"] for row in rows if row.get("label") == label]
        if not label_flops:
            continue
        mins.append(min(label_flops))
        maxs.append(max(label_flops))
    if len(mins) < 3:
        return None, None
    return max(mins), min(maxs)


def nearest_row_for_budget(rows, label, target_flops, max_rel_error=0.25):
    label_rows = [row for row in rows if row.get("label") == label]
    if not label_rows:
        return None
    best = min(
        label_rows,
        key=lambda row: abs(np.log10(row["_train_flops"]) - np.log10(target_flops)),
    )
    rel_error = abs(best["_train_flops"] - target_flops) / target_flops
    return best if rel_error <= max_rel_error else None


def select_isoflop_curves(rows, labels, num_targets):
    low, high = shared_budget_range(rows, labels)
    if low is None or high is None or low >= high:
        return []

    targets = np.geomspace(low, high, num=max(num_targets, 2))
    selected = []
    seen = set()
    for target in targets:
        curve_rows = []
        for label in labels:
            row = nearest_row_for_budget(rows, label, target)
            if row is not None:
                curve_rows.append(row)
        curve_rows.sort(key=lambda row: row["_n_params"])
        if len(curve_rows) < 3:
            continue
        signature = tuple((row["label"], row["_step"]) for row in curve_rows)
        if signature in seen:
            continue
        seen.add(signature)
        actual_budget = float(np.median([row["_train_flops"] for row in curve_rows]))
        selected.append((actual_budget, curve_rows))
    return selected


def quadratic_fit(curve_rows):
    xs = np.log10(np.array([row["_n_params"] for row in curve_rows], dtype=float))
    ys = np.array([row["_train_loss"] for row in curve_rows], dtype=float)
    if len(xs) < 3:
        return None
    coeffs = np.polyfit(xs, ys, deg=2)
    if coeffs[0] <= 0:
        return None
    x_opt = -coeffs[1] / (2 * coeffs[0])
    return coeffs, x_opt


def fit_power_law(xs, ys):
    if len(xs) < 2:
        return None
    log_x = np.log10(np.array(xs, dtype=float))
    log_y = np.log10(np.array(ys, dtype=float))
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    return slope, intercept


def plot_curves_figure(rows, labels, params, args):
    cmap = plt.cm.turbo
    norm = mcolors.LogNorm(
        vmin=min(params.values()),
        vmax=max(params.values()),
    )
    fig = plt.figure(figsize=(18, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[18, 1.4], hspace=0.35, wspace=0.22)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    cax = fig.add_subplot(gs[1, 0:2])

    for label in labels:
        if label not in params:
            continue
        color = cmap(norm(params[label]))
        x_vals, y_vals = series_lookup(rows, label, "_data_tokens")
        if len(x_vals):
            axes[0].plot(x_vals, y_vals, color=color, linewidth=2, alpha=0.95)
        x_vals, y_vals = series_lookup(rows, label, "_train_flops")
        if len(x_vals):
            axes[1].plot(x_vals, y_vals, color=color, linewidth=2, alpha=0.95)

    for ax, xlabel, title in [
        (axes[0], "Data Tokens", "(a) Training Loss vs Data"),
        (axes[1], "Training FLOPs (proxy)", "(b) Training Loss vs FLOPs"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Training Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)

    isoflop_curves = select_isoflop_curves(rows, labels, args.num_isoflop_curves)
    if isoflop_curves:
        flop_colors = plt.cm.Set2(np.linspace(0.05, 0.95, len(isoflop_curves)))
        for color, (budget, curve_rows) in zip(flop_colors, isoflop_curves):
            xs = np.array([row["_n_params"] for row in curve_rows], dtype=float)
            ys = np.array([row["_train_loss"] for row in curve_rows], dtype=float)
            axes[2].scatter(xs, ys, color=color, edgecolor="#333333", s=85, zorder=3)

            fit = quadratic_fit(curve_rows)
            if fit is not None:
                coeffs, _ = fit
                x_grid_log = np.linspace(np.log10(xs.min()), np.log10(xs.max()), 200)
                y_grid = np.polyval(coeffs, x_grid_log)
                axes[2].plot(10 ** x_grid_log, y_grid, color=color, linewidth=2, alpha=0.55)
            else:
                axes[2].plot(xs, ys, color=color, linewidth=1.8, alpha=0.55)

        axes[2].set_xscale("log")
        axes[2].set_xlabel("Parameters")
        axes[2].set_ylabel("Training Loss")
        axes[2].set_title("(c) IsoFLOP Curves")
        axes[2].grid(True, alpha=0.35)
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="#333333",
                markersize=9,
                label=f"{budget:.2e}",
            )
            for color, (budget, _) in zip(flop_colors, isoflop_curves)
        ]
        axes[2].legend(handles=handles, title="FLOPs", fontsize=9, title_fontsize=10, loc="best")
    else:
        axes[2].text(0.5, 0.5, "Not enough overlap\nfor isoFLOP curves", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("(c) IsoFLOP Curves")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_label("Parameters")
    cax.xaxis.set_label_position("bottom")

    fig.suptitle("Scaling Laws in DeepFleet-like Style", fontsize=16, fontweight="bold", y=0.99)
    output_path = Path(args.output_curves)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {output_path}")
    return isoflop_curves


def plot_optimal_figure(isoflop_curves, args):
    optimal_points = []
    for budget, curve_rows in isoflop_curves:
        fit = quadratic_fit(curve_rows)
        if fit is None:
            continue
        coeffs, x_opt = fit
        n_opt = 10 ** x_opt
        d_opt = budget / (args.flop_coeff * n_opt)
        optimal_points.append((budget, d_opt, n_opt, coeffs))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    fig.suptitle("Compute-Optimal Scaling", fontsize=15, fontweight="bold", y=0.98)

    if len(optimal_points) >= 2:
        budgets = np.array([item[0] for item in optimal_points], dtype=float)
        d_opts = np.array([item[1] for item in optimal_points], dtype=float)
        n_opts = np.array([item[2] for item in optimal_points], dtype=float)

        d_fit = fit_power_law(budgets, d_opts)
        n_fit = fit_power_law(budgets, n_opts)
        future_budget = 10 ** (np.floor(np.log10(budgets.max())) + 1)
        x_grid = np.geomspace(budgets.min() * 0.8, future_budget, 300)

        axes[0].scatter(budgets, d_opts, color="#1f77b4", s=85, edgecolor="#333333", zorder=3)
        if d_fit is not None:
            slope, intercept = d_fit
            y_grid = 10 ** intercept * np.power(x_grid, slope)
            axes[0].plot(x_grid, y_grid, "--", color="#888888", linewidth=2.5)
            future_data = 10 ** intercept * future_budget ** slope
            axes[0].axvline(future_budget, color="#888888", linewidth=2.5, alpha=0.85)
            axes[0].axhline(future_data, color="#888888", linewidth=2.5, alpha=0.85)
            axes[0].text(
                0.03,
                0.97,
                f"1e15 FLOPs -> {future_data / 1e6:.2f}M tokens",
                fontsize=10.5,
                va="top",
                ha="left",
                transform=axes[0].transAxes,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#bbbbbb"),
            )

        axes[1].scatter(budgets, n_opts, color="#1f77b4", s=85, edgecolor="#333333", zorder=3)
        if n_fit is not None:
            slope, intercept = n_fit
            y_grid = 10 ** intercept * np.power(x_grid, slope)
            axes[1].plot(x_grid, y_grid, "--", color="#888888", linewidth=2.5)
            future_params = 10 ** intercept * future_budget ** slope
            axes[1].axvline(future_budget, color="#888888", linewidth=2.5, alpha=0.85)
            axes[1].axhline(future_params, color="#888888", linewidth=2.5, alpha=0.85)
            axes[1].text(
                0.03,
                0.97,
                f"1e15 FLOPs -> {future_params / 1e6:.1f}M params",
                fontsize=10.5,
                va="top",
                ha="left",
                transform=axes[1].transAxes,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#bbbbbb"),
            )

        for ax, ylabel, title in [
            (axes[0], "Optimal Data Tokens", "(d) Optimal Data vs FLOPs"),
            (axes[1], "Optimal Parameters", "(e) Optimal Model Size vs FLOPs"),
        ]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("FLOPs")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.35)
    else:
        for ax, title in zip(axes, ["(d) Optimal Data vs FLOPs", "(e) Optimal Model Size vs FLOPs"]):
            ax.text(0.5, 0.5, "Not enough isoFLOP fits", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)

    fig.text(
        0.5,
        0.01,
        f"FLOPs proxy uses C = {args.flop_coeff:g} * N * D, where D is CSV data_tokens.",
        ha="center",
        fontsize=10,
        style="italic",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.77, bottom=0.20, wspace=0.25)
    output_path = Path(args.output_optimal)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    args = parse_args()
    all_rows, valid_rows = load_rows(args.csv)
    labels = ordered_labels(all_rows)
    params = param_map(all_rows, labels)
    rows = enrich_rows(valid_rows, args.flop_coeff)

    if len(rows) < 5:
        raise SystemExit("Not enough valid rows to plot.")

    isoflop_curves = plot_curves_figure(rows, labels, params, args)
    plot_optimal_figure(isoflop_curves, args)


if __name__ == "__main__":
    main()
