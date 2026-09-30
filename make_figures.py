"""Regenerate every figure of the EfficientMatch paper from the files in results/.

Each entry in FIGURES below is one figure, with the paper number it carries, the plotting script it
calls and the exact arguments used to produce it. No training is involved: the figures are read back
from the metrics JSON files, so this runs in seconds and needs no GPU.

    python make_figures.py --list        # what each paper figure is made of
    python make_figures.py               # produce whatever is missing from figures/
    python make_figures.py --figure 2    # just that one
    python make_figures.py --force       # redraw everything, overwriting figures/

Figures already present are skipped, so an accepted figure is never silently overwritten.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
FIGURES = os.path.join(REPO_ROOT, "figures")

# Per-configuration axis limits. The y-axis is cropped to the range where the methods actually
# separate -- showing 0 to 1 would bunch every curve against the top -- and the x-axis is zoomed on
# the methods that converge, so a method that never does simply trails off the right edge.
AXES = {
    "svhn-250":      dict(dataset="svhn",     num_labeled=250,  target_acc=0.90,
                          xmax={"time": "45", "steps": "43500", "flops": "37000000"},
                          opts=["--ymin", "0.75", "--auto_ymax", "--legend_top", "--auto_xmin"]),
    "cifar10-250":   dict(dataset="cifar10",  num_labeled=250,  target_acc=0.80,
                          opts=["--ymin", "0.60", "--ymax", "0.85", "--auto_xmax",
                                "--legend_loc", "best"]),
    "cifar10-4000":  dict(dataset="cifar10",  num_labeled=4000, target_acc=0.90,
                          opts=["--ymin", "0.85", "--auto_ymax", "--legend_top", "--auto_xmax",
                                "--auto_xmin"]),
    "cifar100-2500": dict(dataset="cifar100", num_labeled=2500, target_acc=0.50, widen_factor=4,
                          opts=["--ymin", "0.30", "--auto_ymax", "--legend_top", "--auto_xmax",
                                "--auto_xmin"]),
}


def all_methods(config, seed, xaxis):
    """One panel of the five compared methods, accuracy against iterations, time or FLOPs."""
    cfg = AXES[config]
    args = ["plot_acc_vs_time.py", "--dataset", cfg["dataset"],
            "--num_labeled", str(cfg["num_labeled"]), "--seed", str(seed),
            "--target_acc", str(cfg["target_acc"]), "--xaxis", xaxis] + cfg["opts"]
    if "xmax" in cfg:   # a fixed x limit, one per axis since each has its own unit
        args += ["--xmax", cfg["xmax"][xaxis]]
    if cfg.get("widen_factor"):
        args += ["--widen_factor", str(cfg["widen_factor"])]
    name = f"all_methods_acc_vs_{xaxis}_{cfg['dataset']}_{cfg['num_labeled']}_seed{seed}.png"
    return name, args


def build_figures():
    figs = []

    # Figure 1 -- model accuracy against the quality of the pseudo-labels it is trained on, one
    # panel per method, showing that EfficientMatch inherits both FixMatch's pseudo-label quality
    # and MixMatch's gain over it.
    for method in ("fixmatch", "mixmatch", "efficientmatch"):
        figs.append(dict(paper="Figure 1", name=f"{method}_acc_vs_pl_cifar10_250_seed2701.png",
                         args=["plot_acc_vs_pl.py", "--dataset", "cifar10", "--num_labeled", "250",
                               "--seed", "2701", "--methods", method]))

    # Figure 2 -- the same run ranked by each of the three criteria, which disagree.
    for xaxis in ("steps", "time", "flops"):
        name, args = all_methods("cifar10-250", 2701, xaxis)
        figs.append(dict(paper="Figure 2", name=name, args=args))

    # Figure 3 -- weight of the Mixup loss: accuracy, then pseudo-label quality per weight.
    for xaxis in ("time", "steps"):
        figs.append(dict(paper="Figure 3", name=f"ablation_mixw_acc_vs_{xaxis}_cifar10_250_seed2312.png",
                         args=["plot_ablation_mixw.py"]))
    for weight in ("0.5", "1", "2"):
        figs.append(dict(paper="Figure 3", name=f"ablation_mixw{weight}_acc_vs_pl_cifar10_250_seed2312.png",
                         args=["plot_ablation_mixw_pl.py", "2312"]))

    # Figure 4 -- accuracy against time on all four configurations, same seed.
    for config in AXES:
        name, args = all_methods(config, 2312, "time")
        figs.append(dict(paper="Figure 4", name=name, args=args))

    # Figure 5 -- the same configuration on the three seeds, showing how much the ranking moves.
    for seed in (2312, 308, 2701):
        name, args = all_methods("cifar10-250", seed, "time")
        figs.append(dict(paper="Figure 5", name=name, args=args))

    # Supporting figures: the other three configurations by iterations and FLOPs, the seeds Figures
    # 4 and 5 do not show, and the extended-horizon comparison behind Table 11.
    for config in ("svhn-250", "cifar10-4000", "cifar100-2500"):
        for xaxis in ("steps", "flops"):
            name, args = all_methods(config, 2701, xaxis)
            figs.append(dict(paper="supporting", name=name, args=args))
        for seed in (308, 2701):
            name, args = all_methods(config, seed, "time")
            figs.append(dict(paper="supporting", name=name, args=args))
    for xaxis in ("steps", "time", "flops"):
        figs.append(dict(
            paper="Table 11", name=f"unlimited_efficientmatch_vs_regmixmatch_acc_vs_{xaxis}_cifar10_250_seed42.png",
            args=["plot_acc_vs_time.py", "--dataset", "cifar10", "--num_labeled", "250", "--seed", "42",
                  "--target_acc", "0.9", "--xaxis", xaxis, "--ymin", "0.70", "--ymax", "0.93",
                  "--no_budget", "--hide_target", "--legend_loc", "lower right", "--methods",
                  "efficientmatch:efficientmatch_ema_unlimited_metrics.json,"
                  "regmixmatch:regmixmatch_ema_unlimited_metrics.json",
                  "--out", os.path.join(FIGURES,
                      f"unlimited_efficientmatch_vs_regmixmatch_acc_vs_{xaxis}_cifar10_250_seed42.png")]))

    # The summary at the top of the README: minutes to target, every method, configuration and seed.
    figs.append(dict(paper="README", name="summary_time_to_target.png", args=["plot_summary.py"]))

    # Deduplicate: Figure 2 and Figure 5 share the CIFAR-10/250 seed 2701 panel, and Figure 4
    # shares its CIFAR-10/250 panel with Figure 5 seed 2312.
    seen, unique = set(), []
    for f in figs:
        if f["name"] in seen:
            continue
        seen.add(f["name"])
        unique.append(f)
    return unique


def main():
    figs = build_figures()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", help="Only this one, e.g. 2 or 'Figure 2'.")
    ap.add_argument("--force", action="store_true", help="Redraw figures that already exist.")
    ap.add_argument("--list", action="store_true", help="List the figures and their commands.")
    args = ap.parse_args()

    if args.figure:
        wanted = args.figure if args.figure.lower().startswith(("figure", "table", "readme")) else f"Figure {args.figure}"
        figs = [f for f in figs if f["paper"].lower() == wanted.lower()]
        if not figs:
            sys.exit(f"No figure matches {args.figure!r}.")

    if args.list:
        for f in figs:
            mark = "ok     " if os.path.exists(os.path.join(FIGURES, f["name"])) else "missing"
            print(f"{mark} {f['paper']:12s}{f['name']}")
            print(f"            python scripts/{' '.join(f['args'])}")
        return

    todo = [f for f in figs if args.force or not os.path.exists(os.path.join(FIGURES, f["name"]))]
    if not todo:
        print(f"All {len(figs)} figures are already in figures/ (use --force to redraw).")
        return

    # Several scripts draw more than one file per call, so run each distinct command once.
    commands, order = {}, []
    for f in todo:
        key = tuple(f["args"])
        if key not in commands:
            commands[key] = f["paper"]
            order.append(key)
    for key in order:
        cmd = [sys.executable] + [os.path.join(SCRIPTS, key[0])] + list(key[1:])
        if args.force and key[0] in ("plot_acc_vs_pl.py", "plot_ablation_mixw_pl.py"):
            cmd.append("--overwrite")
        print(f"[{commands[key]}] {' '.join(key)}")
        result = subprocess.run(cmd, cwd=SCRIPTS)
        if result.returncode != 0:
            sys.exit(f"Figure command failed ({result.returncode}).")


if __name__ == "__main__":
    main()
