"""Accuracy curves of the mixup_weight ablation (efficientmatch_3, CIFAR-10 250 labels)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = [("efficientmatch_3_ema_mixw0.5", r"$\lambda_{mix}$=0.5"), ("efficientmatch_3_ema", r"$\lambda_{mix}$=1"),
            ("efficientmatch_3_ema_mixw2.0", r"$\lambda_{mix}$=2")]
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9})
for seed in (2312, 308, 2701):
    d = os.path.join(ROOT, "results", f"cifar10-labeled-250-seed-{seed}")
    runs = [(lab, json.load(open(os.path.join(d, f"{f}_metrics.json")))) for f, lab in VARIANTS]
    for axis in ("time", "steps"):
        out = os.path.join(ROOT, "figures", f"ablation_mixw_acc_vs_{axis}_cifar10_250_seed{seed}.png")
        if os.path.exists(out):
            print("skip (exists)", out); continue
        fig, ax = plt.subplots(figsize=(3.4, 3.1))
        for lab, j in runs:
            x = [v / 60 for v in j["time_elapsed"]] if axis == "time" else [v / 1000 for v in j["step"]]
            ax.plot(x, [100 * v for v in j["test_acc"]], label=lab, lw=1.3)
        ax.set_ylim(60, 85)
        ax.set_xlabel("Time (min)" if axis == "time" else "Steps (k)"); ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Seed {seed}", fontweight="bold")
        ax.grid(alpha=0.3, lw=0.5); ax.legend(loc="lower right", frameon=True, framealpha=0.85)
        fig.tight_layout(pad=0.4); fig.savefig(out, dpi=300); plt.close(fig); print("saved", out)
