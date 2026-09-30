"""Summary figure for the README: wall-clock minutes to reach the target accuracy, per method and
configuration, corrected for evaluation cost exactly as in Tables 2 and 3.

Each bar spans the three seeds, from the fastest to the slowest, and each dot is one seed. Like
the paper, it reports no average: the seed moves the results too much for one to be meaningful. A
method that never reaches the target within the two-hour budget gets no bar, only a dagger, as in
the paper's tables.

    python scripts/plot_summary.py          # writes figures/summary_time_to_target.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_analysis import EVAL_SECONDS, corrected_minutes, first_reaching  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "summary_time_to_target.png")

CONFIGS = [  # (dataset, labels, target, widen factor, panel title)
    ("svhn", 250, 0.90, 2, "SVHN, 250 labels"),
    ("cifar10", 250, 0.80, 2, "CIFAR-10, 250 labels"),
    ("cifar10", 4000, 0.90, 2, "CIFAR-10, 4000 labels"),
    ("cifar100", 2500, 0.50, 4, "CIFAR-100, 2500 labels"),
]
SEEDS = (2312, 308, 2701)
METHODS = [("efficientmatch", "EfficientMatch"), ("fixmatch", "FixMatch"), ("flexmatch", "FlexMatch"),
           ("mixmatch", "MixMatch"), ("regmixmatch", "RegMixMatch")]
BUDGET_S = 120 * 60

HIGHLIGHT = "#1f77b4"   # EfficientMatch's colour in every figure of the paper
OTHERS = "#c4c4c4"
INK, MUTED = "#222222", "#6b6b6b"


def minutes_to_target(dataset, labels, target, wf, method, seed):
    """Corrected minutes to the first evaluation at or above target, or None if not reached in 2 h."""
    name = f"{method}_ema" + (f"_wf{wf}" if wf != 2 else "")
    path = os.path.join(ROOT, "results", f"{dataset}-labeled-{labels}-seed-{seed}", f"{name}_metrics.json")
    with open(path) as fh:
        data = json.load(fh)
    hit = first_reaching(data, target)
    if hit is None or hit[3] > BUDGET_S:
        return None
    idx, _, _, time_s = hit
    return corrected_minutes(time_s, idx + 1, EVAL_SECONDS[wf])


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
    y = list(range(len(METHODS)))[::-1]   # EfficientMatch on top

    for ax, (dataset, labels, target, wf, title) in zip(axes, CONFIGS):
        values = {m: [minutes_to_target(dataset, labels, target, wf, m, s) for s in SEEDS] for m, _ in METHODS}
        xmax = max(v for vs in values.values() for v in vs if v is not None) * 1.38
        for yi, (method, _) in zip(y, METHODS):
            vs = values[method]
            color = HIGHLIGHT if method == "efficientmatch" else OTHERS
            if any(v is None for v in vs):
                ax.text(xmax * 0.02, yi, "† not reached in 2 h", va="center", ha="left", color=MUTED,
                        fontsize=8.5, style="italic")
                continue
            lo, hi = min(vs), max(vs)
            ax.barh(yi, hi - lo, left=lo, height=0.62, color=color, zorder=2)
            ax.scatter(vs, [yi] * len(vs), s=14, color=INK, zorder=3, linewidths=0)
            ax.text(hi + xmax * 0.025, yi, (f"{lo:.1f}–{hi:.1f}" if hi < 20 else f"{lo:.0f}–{hi:.0f}"), va="center", ha="left", color=INK, fontsize=9,
                    fontweight="bold" if method == "efficientmatch" else "normal")
        ax.set_xlim(0, xmax)
        ax.set_title(f"{title}\ntarget {target:.0%}", fontweight="bold", fontsize=10.5, color=INK)
        ax.set_xlabel("minutes", color=MUTED)
        ax.grid(axis="x", alpha=0.3, linewidth=0.5, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=MUTED)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label for _, label in METHODS], color=INK)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    fig.text(0.995, 0.01, "bar: range over seeds 2312, 0308, 2701 · dot: one seed · time corrected for evaluation cost",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=1.2)
    fig.savefig(OUT, dpi=200)
    print("saved to", os.path.relpath(OUT, ROOT))
    for dataset, labels, target, wf, title in CONFIGS:   # the numbers drawn, to check against Table 2
        print(title, {m: [None if v is None else round(v, 1) for v in
                          (minutes_to_target(dataset, labels, target, wf, m, s) for s in SEEDS)] for m, _ in METHODS})


if __name__ == "__main__":
    main()
