"""Model accuracy vs pseudo-label quality, same style as the original fixmatch/mixmatch acc_vs_pl figures."""
import json, os, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--dataset", default="cifar10"); p.add_argument("--num_labeled", type=int, default=250)
p.add_argument("--seed", type=int, default=2701)
p.add_argument("--methods", nargs="+", default=["efficientmatch_3"])
p.add_argument("--ymin", type=float, default=50); p.add_argument("--ymax", type=float, default=84)
a = p.parse_args()

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})
d = os.path.join(ROOT, "results", f"{a.dataset}-labeled-{a.num_labeled}-seed-{a.seed}")
for m in a.methods:
    j = json.load(open(os.path.join(d, f"{m}_ema_metrics.json")))
    t = [x / 60 for x in j["time_elapsed"]]
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    ax.plot(t, [100 * v for v in j["test_acc"]], label="Model accuracy", lw=1.5)
    ax.plot(t, [100 * v for v in j["pl_quality"]], label="Pseudo-label quality", lw=1.5)
    ax.set_ylim(a.ymin, a.ymax)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Accuracy (%)")
    name = "efficientmatch" if m == "efficientmatch_3" else m
    ax.set_title(name, fontweight="bold")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.85)
    fig.tight_layout(pad=0.4)
    out = os.path.join(ROOT, "figures", f"{name}_acc_vs_pl_{a.dataset}_{a.num_labeled}_seed{a.seed}.png")
    if os.path.exists(out):
        print("skip (exists)", out); continue
    fig.savefig(out, dpi=300); print("saved", out)
