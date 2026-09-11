import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Metrics tracked for now; extend this list to bring more metrics back into the
# table/plots without touching the rest of the script.
TRACKED_METRICS = ["test_acc", "mask_ratio", "pl_quality", "reinforcement_ratio", "corrections_ratio"]

# name -> (numerator field, denominator field) for ratios derived from raw counts in the metrics json.
DERIVED_RATIOS = {
    "reinforcement_ratio": ("correct_reinforcement", "error_reinforcement"),
    "corrections_ratio": ("corrections", "bad_corrections"),
}

# FLOPs per training iteration (WideResNet-28-2, batch labellisé=64), depuis FLOPS_RESULTS.md /
# `scripts/flops_analysis.py`. Utilisé pour tracer test_acc en fonction des FLOPs cumulés.
FLOPS_PER_ITER = {
    "efficientmatch": 7.404e11,
    "fixmatch": 8.501e11,
    "flexmatch": 8.501e11,
    "mixmatch": 3.016e11,
    "sequencematch": 1.810e12,
}


def infer_flops_per_iter(label):
    for name, flops in FLOPS_PER_ITER.items():
        if label.lower().startswith(name):
            return flops
    return None


def load_metrics(metrics_path, test_period):
    with open(metrics_path) as f:
        metrics = json.load(f)

    needed_cols = set(TRACKED_METRICS) | {"time_elapsed"}
    for num_col, den_col in DERIVED_RATIOS.values():
        needed_cols.update([num_col, den_col])

    df = pd.DataFrame({k: v for k, v in metrics.items() if k in needed_cols})
    for ratio_name, (num_col, den_col) in DERIVED_RATIOS.items():
        if num_col in df.columns and den_col in df.columns:
            df[ratio_name] = df[num_col] / df[den_col].replace(0, float("nan"))

    df = df[[c for c in TRACKED_METRICS + ["time_elapsed"] if c in df.columns]]
    # Evaluations happen at step 0, then every test_period steps (see the training loop's
    # "(step + 1) % test_period == 0 or step == 0" condition), so reconstruct the step index.
    n = len(df)
    df.insert(0, "step", [0] + [i * test_period for i in range(1, n)])
    return df


def print_summary_table(df):
    cols = ["step"] + [c for c in TRACKED_METRICS if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    logger.info(
        f"Best test_acc: {df['test_acc'].max():.4f} at step {df.loc[df['test_acc'].idxmax(), 'step']}, "
        f"Final test_acc: {df['test_acc'].iloc[-1]:.4f}"
    )


def make_plots(df, output_dir, method_name, flops_per_iter=None):
    os.makedirs(output_dir, exist_ok=True)

    cols = [c for c in TRACKED_METRICS if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), squeeze=False)

    for ax, col in zip(axes[0], cols):
        ax.plot(df["step"], df[col])
        ax.set_title(col)
        ax.set_xlabel("step")

    fig.suptitle(method_name)
    fig.tight_layout()
    overview_path = os.path.join(output_dir, f"{method_name}_overview.png")
    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {overview_path}")

    if "time_elapsed" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["time_elapsed"], df["test_acc"])
        ax.set_title(f"{method_name} - test_acc vs time")
        ax.set_xlabel("time elapsed (s)")
        ax.set_ylabel("test_acc")
        fig.tight_layout()
        time_path = os.path.join(output_dir, f"{method_name}_acc_vs_time.png")
        fig.savefig(time_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {time_path}")

    if flops_per_iter is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["step"] * flops_per_iter, df["test_acc"])
        ax.set_title(f"{method_name} - test_acc vs FLOPs")
        ax.set_xlabel("cumulative FLOPs")
        ax.set_ylabel("test_acc")
        fig.tight_layout()
        flops_path = os.path.join(output_dir, f"{method_name}_acc_vs_flops.png")
        fig.savefig(flops_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {flops_path}")


def print_comparison_table(runs):
    rows = []
    for name, df in runs.items():
        best_idx = df["test_acc"].idxmax()
        row = {
            "method": name,
            "best_test_acc": df["test_acc"].max(),
            "best_step": df.loc[best_idx, "step"],
            "final_test_acc": df["test_acc"].iloc[-1],
            "steps_covered": df["step"].iloc[-1],
        }
        for col in ("mask_ratio", "pl_quality", "reinforcement_ratio", "corrections_ratio"):
            row[f"final_{col}"] = df[col].iloc[-1] if col in df.columns else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("best_test_acc", ascending=False)
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summary


def make_comparison_plots(runs, output_dir, flops_per_iter=None):
    os.makedirs(output_dir, exist_ok=True)

    cols = [c for c in TRACKED_METRICS if all(c in df.columns for df in runs.values())]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), squeeze=False)

    for ax, col in zip(axes[0], cols):
        for name, df in runs.items():
            ax.plot(df["step"], df[col], label=name)
        ax.set_title(col)
        ax.set_xlabel("step")
        ax.legend()

    fig.suptitle("Method comparison")
    fig.tight_layout()
    path = os.path.join(output_dir, "comparison_overview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")

    if all("time_elapsed" in df.columns for df in runs.values()):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, df in runs.items():
            ax.plot(df["time_elapsed"], df["test_acc"], label=name)
        ax.set_title("test_acc vs time")
        ax.set_xlabel("time elapsed (s)")
        ax.set_ylabel("test_acc")
        ax.legend()
        fig.tight_layout()
        time_path = os.path.join(output_dir, "comparison_acc_vs_time.png")
        fig.savefig(time_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {time_path}")

    if flops_per_iter and all(name in flops_per_iter for name in runs):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, df in runs.items():
            ax.plot(df["step"] * flops_per_iter[name], df["test_acc"], label=name)
        ax.set_title("test_acc vs FLOPs")
        ax.set_xlabel("cumulative FLOPs")
        ax.set_ylabel("test_acc")
        ax.legend()
        fig.tight_layout()
        flops_path = os.path.join(output_dir, "comparison_acc_vs_flops.png")
        fig.savefig(flops_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {flops_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_paths", type=str, nargs="+", help="One or more *_metrics.json files. Pass several to compare methods.")
    parser.add_argument("--labels", type=str, nargs="+", default=None, help="Custom labels for each metrics_path (default: inferred from filename)")
    parser.add_argument("--test_period", type=int, default=500, help="test_period used for the run(s) (for the step axis)")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save plots/CSV (default: alongside the metrics file, or 'results/comparison' when comparing)")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.metrics_paths):
        parser.error("--labels must have the same number of entries as metrics_paths")

    labels = args.labels or [os.path.basename(p).replace("_metrics.json", "") for p in args.metrics_paths]

    runs = {}
    flops_per_iter = {}
    for path, label in zip(args.metrics_paths, labels):
        df = load_metrics(path, args.test_period)
        runs[label] = df
        label_flops = infer_flops_per_iter(label)
        if label_flops is not None:
            flops_per_iter[label] = label_flops

        output_dir = args.output_dir or os.path.join(os.path.dirname(path), "plots")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{label}_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")
        make_plots(df, output_dir, label, flops_per_iter=label_flops)

    if len(runs) == 1:
        (df,) = runs.values()
        print_summary_table(df)
    else:
        comparison_dir = args.output_dir or "results/comparison"
        summary = print_comparison_table(runs)
        os.makedirs(comparison_dir, exist_ok=True)
        summary_path = os.path.join(comparison_dir, "comparison_summary.csv")
        summary.to_csv(summary_path, index=False)
        logger.info(f"Saved {summary_path}")
        make_comparison_plots(runs, comparison_dir, flops_per_iter=flops_per_iter)


if __name__ == "__main__":
    main()
