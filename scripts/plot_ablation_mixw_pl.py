"""Model accuracy vs pseudo-label quality for each mixup_weight of the ablation (CIFAR-10 250 labels)."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2312
VARIANTS = [("efficientmatch_3_ema_mixw0.5", "0.5"), ("efficientmatch_3_ema", "1"), ("efficientmatch_3_ema_mixw2.0", "2")]
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8})
d = os.path.join(ROOT, "results", f"cifar10-labeled-250-seed-{seed}")
for f, w in VARIANTS:
    out = os.path.join(ROOT, "figures", f"ablation_mixw{w}_acc_vs_pl_cifar10_250_seed{seed}.png")
    if os.path.exists(out):
        print("skip (exists)", out); continue
    j = json.load(open(os.path.join(d, f"{f}_metrics.json")))
    t = [x / 60 for x in j["time_elapsed"]]
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    ax.plot(t, [100 * v for v in j["test_acc"]], label="Model accuracy", lw=1.5)
    ax.plot(t, [100 * v for v in j["pl_quality"]], label="Pseudo-label quality", lw=1.5)
    ax.set_ylim(50, 84)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(rf"$\lambda_{{mix}}$ = {w}", fontweight="bold")
    ax.grid(alpha=0.3, lw=0.5); ax.legend(loc="lower right", frameon=True, framealpha=0.85)
    fig.tight_layout(pad=0.4); fig.savefig(out, dpi=300); plt.close(fig); print("saved", out)
