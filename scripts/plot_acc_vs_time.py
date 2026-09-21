"""Reusable accuracy-vs-{time,steps,flops} plotting tool for conference-style figures on this
project.

Reads metrics JSON files directly (no training code involved) and plots test_acc against one of
three x-axes for one or more methods on a given config, styled for a single-column conference
figure (small figsize, large fonts, serif, 300dpi -- see memory/publication-figure-style.md
conventions).

By default, the 5 core methods (efficientmatch_3, fixmatch, flexmatch, mixmatch, regmixmatch)
are plotted using their standard "<method>_ema_metrics.json" (or "..._ema_wf4_metrics.json" for
cifar100) file. Use --methods to override with an explicit "label:filename" list when a config
has ambiguous files (e.g. fixmatch_ema vs fixmatch_ema_orig vs fixmatch_ema_bis).

"efficientmatch_3" is always displayed as "efficientmatch" in the legend (the "_3" is an internal
variant label, not something readers of the figure need) -- this applies whether the method comes
from the default list or from --methods.

Usage:
    python plot_acc_vs_time.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80
    python plot_acc_vs_time.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 \
        --xaxis steps --xmax 40000
    python plot_acc_vs_time.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 \
        --xaxis flops
    python plot_acc_vs_time.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 \
        --xmax 120 --ymin 0 --ymax 1.0 --legend_loc "lower right"
    python plot_acc_vs_time.py --dataset cifar100 --num_labeled 10000 --seed 2312 --target_acc 0.60 \
        --methods "efficientmatch_3:efficientmatch_3_ema_wf4_metrics.json,fixmatch:fixmatch_ema_wf4_metrics.json"
    python plot_acc_vs_time.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 \
        --methods "efficientmatch_3:efficientmatch_3_ema_metrics.json,fixmatch:fixmatch_ema_bis_metrics.json"

Output is saved directly into the repo's figures/ directory (default naming:
all_methods_acc_vs_<xaxis>_<dataset>_<num_labeled>_seed<seed>.png), matching the standing
convention of keeping every validated figure there rather than only in a scratchpad.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
FIGURES_ROOT = os.path.join(REPO_ROOT, "figures")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_analysis import gflops_for  # noqa: E402

DEFAULT_METHOD_ORDER = ["efficientmatch_3", "fixmatch", "flexmatch", "mixmatch", "regmixmatch"]

# Cosmetic-only renames applied to whatever label is used (default or --methods-supplied).
DISPLAY_LABEL_OVERRIDES = {"efficientmatch_3": "efficientmatch"}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "serif",
})


def display_label(label):
    return DISPLAY_LABEL_OVERRIDES.get(label, label)


def default_methods(dataset):
    suffix = "_wf4" if dataset == "cifar100" else ""
    return [(label, f"{label}_ema{suffix}_metrics.json") for label in DEFAULT_METHOD_ORDER]


def parse_methods_arg(methods_arg):
    pairs = []
    for item in methods_arg.split(","):
        label, fname = item.split(":", 1)
        pairs.append((label.strip(), fname.strip()))
    return pairs


def dataset_title(dataset):
    return {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "svhn": "SVHN"}.get(dataset, dataset)


AXIS_CONFIG = {
    "time": {"label": "Temps (min)", "suffix": "time"},
    "steps": {"label": "Iterations", "suffix": "steps"},
    "flops": {"label": "GFLOPs (cumulés)", "suffix": "flops"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_labeled", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target_acc", type=float, required=True, help="Target accuracy (0-1 fraction) for the horizontal line.")
    ap.add_argument("--methods", type=str, default=None, help='Override as "label:filename,label:filename,...". Defaults to the 5 core methods.')
    ap.add_argument("--xaxis", choices=["time", "steps", "flops"], default="time", help="What to plot accuracy against.")
    ap.add_argument("--widen_factor", type=int, default=None, help="Architecture for FLOPs lookup. Defaults to 4 for cifar100, 2 otherwise.")
    ap.add_argument("--xmax", type=float, default=None, help="Max x-axis. Defaults to the longest run's last value on that axis.")
    ap.add_argument("--ymin", type=float, default=0.0)
    ap.add_argument("--ymax", type=float, default=1.0)
    ap.add_argument("--legend_loc", type=str, default="lower right")
    ap.add_argument("--out", type=str, default=None, help="Output path. Defaults to figures/all_methods_acc_vs_<xaxis>_<dataset>_<num_labeled>_seed<seed>.png")
    args = ap.parse_args()

    result_dir = os.path.join(RESULTS_ROOT, f"{args.dataset}-labeled-{args.num_labeled}-seed-{args.seed}")
    methods = parse_methods_arg(args.methods) if args.methods else default_methods(args.dataset)
    wf = args.widen_factor if args.widen_factor is not None else (4 if args.dataset == "cifar100" else 2)

    fig, ax = plt.subplots(figsize=(3.4, 3.1))

    xmax = 0
    for label, fname in methods:
        path = os.path.join(result_dir, fname)
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping {label}")
            continue
        with open(path) as f:
            d = json.load(f)
        acc = d["test_acc"]

        if args.xaxis == "time":
            xvals = [t / 60 for t in d["time_elapsed"]]
        elif args.xaxis == "steps":
            xvals = d["step"]
        else:
            gflops_per_it = gflops_for(label, wf)
            if gflops_per_it is None:
                print(f"WARNING: no GFLOPs/iteration entry for '{label}' at wf={wf}, skipping")
                continue
            xvals = [s * gflops_per_it for s in d["step"]]

        xmax = max(xmax, xvals[-1])
        ax.plot(xvals, acc, label=display_label(label), linewidth=1.3)

    ax.axhline(args.target_acc, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label=f"target {args.target_acc*100:.0f}%")

    ax.set_xlabel(AXIS_CONFIG[args.xaxis]["label"])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset_title(args.dataset)}, {args.num_labeled} labels\nseed {args.seed}", fontweight="bold", fontsize=11, pad=4)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlim(0, args.xmax if args.xmax is not None else xmax)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc=args.legend_loc, frameon=True, framealpha=0.85, fontsize=8)

    fig.tight_layout(pad=0.5)

    suffix = AXIS_CONFIG[args.xaxis]["suffix"]
    out_path = args.out or os.path.join(FIGURES_ROOT, f"all_methods_acc_vs_{suffix}_{args.dataset}_{args.num_labeled}_seed{args.seed}.png")
    fig.savefig(out_path, dpi=300)
    print("saved to", out_path)


if __name__ == "__main__":
    main()
