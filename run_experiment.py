"""Run any experiment reported in the EfficientMatch paper, with a single command.

Every run in the paper is listed in EXPERIMENTS below, together with the paper table it feeds and
the result file it writes. Nothing here re-implements training: each entry is turned into a call to
the matching script in scripts/, with the exact arguments used for the paper.

    python run_experiment.py --list                     # what can be run, and what is already done
    python run_experiment.py --check                    # verify every paper run has its result file
    python run_experiment.py --experiment main --dry-run
    python run_experiment.py --experiment main --config cifar10-250 --method efficientmatch --seed 2312

Runs whose result file already exists are skipped, so an interrupted sweep can simply be relaunched;
pass --force to recompute them. Reproducing a whole experiment takes a long time: the main sweep is
60 runs, each capped at 2 hours.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
RESULTS = os.path.join(REPO_ROOT, "results")

# The four dataset/label-budget combinations of the paper (§4). target_acc is the pre-asymptotic
# threshold each run stops at; widen_factor is the WideResNet width (WRN-28-2, or WRN-28-4 on
# CIFAR-100, which the 8GB GPU used for the paper cannot run at width 8).
CONFIGS = {
    "svhn-250":      dict(dataset="svhn",     num_labeled=250,  target_acc=0.90, widen_factor=2),
    "cifar10-250":   dict(dataset="cifar10",  num_labeled=250,  target_acc=0.80, widen_factor=2),
    "cifar10-4000":  dict(dataset="cifar10",  num_labeled=4000, target_acc=0.90, widen_factor=2),
    "cifar100-2500": dict(dataset="cifar100", num_labeled=2500, target_acc=0.50, widen_factor=4),
}

SEEDS = (2312, 308, 2701)

# Each method keeps the unlabeled:labeled ratio mu established for it in the literature (§3), which
# is the default in its own script -- so mu never has to be passed here.
MAIN_METHODS = ("efficientmatch", "fixmatch", "flexmatch", "mixmatch", "regmixmatch")

# Wall-clock cap of the paper's stopping criterion (§4): a run stops at target_acc or at 2 hours.
BUDGET_MINUTES = 120


def result_file(config, script, seed, pre="", post=""):
    """Path the run writes its metrics to, mirroring the method_name each script builds:
    <script><pre>_ema<post>[_wf<N>], with _wf<N> omitted at the default width 2. Whether a variant
    suffix lands before or after _ema depends on the script, hence the two explicit slots: FixMatch
    and RegMixMatch put _mu<N> before _ema, EfficientMatch puts _mu<N>/_mixw<W> after it."""
    cfg = CONFIGS[config]
    wf = "" if cfg["widen_factor"] == 2 else f"_wf{cfg['widen_factor']}"
    run_dir = f"{cfg['dataset']}-labeled-{cfg['num_labeled']}-seed-{seed}"
    return os.path.join(RESULTS, run_dir, f"{script}{pre}_ema{post}{wf}_metrics.json")


def run(experiment, paper, config, script, seed, extra=(), pre="", post="", budget=True):
    return dict(experiment=experiment, paper=paper, config=config, script=script, seed=seed,
                extra=list(extra), file=result_file(config, script, seed, pre, post), budget=budget)


def build_experiments():
    """Every run behind a number in the paper, grouped by the experiment it belongs to."""
    runs = []

    # Tables 2, 3, 12, 13 -- the five methods compared on the four configurations.
    for config in CONFIGS:
        for method in MAIN_METHODS:
            for seed in SEEDS:
                runs.append(run("main", "Tables 2, 3, 12, 13", config, method, seed))

    # Table 4 and §6 -- EfficientMatch with FreeMatch's self-adaptive thresholding.
    for config in CONFIGS:
        for seed in SEEDS:
            runs.append(run("freematch", "Table 4", config, "efficientmatch", seed,
                            extra=["--freematch_threshold", "True"], pre="_freematch"))

    # Appendix A.5 -- the same variant on SVHN without the [0.9, 0.95] clamp, which never converges.
    runs.append(run("freematch-noclamp", "Appendix A.5", "svhn-250", "efficientmatch", 2312,
                    extra=["--freematch_threshold", "True", "--freematch_svhn_clamp", "False"],
                    pre="_freematch_noclamp"))
    runs.append(run("freematch-noclamp", "Appendix A.5", "svhn-250", "regmixmatch", 2312,
                    extra=["--svhn_clamp", "False"], pre="_noclamp"))

    # Tables 5, 6 -- unlabeled ratio mu. mu=3 is the default and already covered by "main".
    for config in ("cifar10-250", "cifar10-4000"):
        for mu in (1, 5, 7):
            for seed in SEEDS:
                runs.append(run("mu", "Tables 5, 6", config, "efficientmatch", seed,
                                extra=["--mu", str(mu)], post=f"_mu{mu}"))

    # Table 7 -- weight of the Mixup loss. lambda_mix=1 is the default, covered by "main".
    for weight in ("0.5", "2.0"):
        for seed in SEEDS:
            runs.append(run("lambda-mix", "Table 7", "cifar10-250", "efficientmatch", seed,
                            extra=["--mixup_weight", weight], post=f"_mixw{weight}"))

    # Table 8 -- what label the Mixup channel is trained against. Semi-soft is the default.
    for script in ("efficientmatch_hard", "efficientmatch_soft"):
        for config in ("cifar10-250", "svhn-250"):
            for seed in SEEDS:
                runs.append(run("mixing-target", "Table 8", config, script, seed))

    # Table 10 -- FixMatch and RegMixMatch brought down to EfficientMatch's mu=3.
    for seed in SEEDS:
        runs.append(run("matched-mu", "Table 10", "cifar10-250", "fixmatch", seed,
                        extra=["--mu", "3"], pre="_mu3"))
        runs.append(run("matched-mu", "Table 10", "cifar10-250", "regmixmatch", seed,
                        extra=["--mu", "3", "--static_shapes", "True"], pre="_mu3"))

    # Table 11 and Appendix A.8 -- no accuracy target, full 2^20-iteration horizon, seed 42.
    # These deliberately ignore both the target and the 2-hour cap: over 12 hours each.
    for script in ("efficientmatch", "regmixmatch"):
        runs.append(run("asymptotic", "Table 11", "cifar10-250", script, 42,
                        extra=["--tag", "unlimited"], post="_unlimited", budget=False))

    return runs


def command(entry):
    """The exact command line this run corresponds to."""
    cfg = CONFIGS[entry["config"]]
    cmd = [sys.executable, os.path.join(SCRIPTS, entry["script"] + ".py"),
           "--dataset", cfg["dataset"], "--num_labeled", str(cfg["num_labeled"]),
           "--seed", str(entry["seed"])]
    if entry["budget"]:
        cmd += ["--target_acc", str(cfg["target_acc"]), "--max_minutes", str(BUDGET_MINUTES)]
    if cfg["widen_factor"] != 2:
        cmd += ["--widen_factor", str(cfg["widen_factor"])]
    return cmd + entry["extra"]


def selected(args, runs):
    out = []
    for e in runs:
        if args.experiment and e["experiment"] != args.experiment:
            continue
        if args.config and e["config"] != args.config:
            continue
        if args.method and e["script"] != args.method:
            continue
        if args.seed and e["seed"] != args.seed:
            continue
        out.append(e)
    return out


def main():
    runs = build_experiments()
    experiments = sorted({e["experiment"] for e in runs})

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=experiments, help="Which paper experiment to run.")
    ap.add_argument("--config", choices=sorted(CONFIGS), help="Restrict to one dataset/label budget.")
    ap.add_argument("--method", help="Restrict to one script, e.g. efficientmatch.")
    ap.add_argument("--seed", type=int, help="Restrict to one seed.")
    ap.add_argument("--force", action="store_true", help="Recompute runs that already have a result file.")
    ap.add_argument("--dry-run", action="store_true", help="Print the commands instead of running them.")
    ap.add_argument("--list", action="store_true", help="List the experiments and how complete each is.")
    ap.add_argument("--check", action="store_true", help="Report which paper runs have no result file.")
    args = ap.parse_args()

    if args.list:
        print(f"{'experiment':20s}{'paper':22s}{'runs':>6s}{'done':>6s}")
        print("-" * 54)
        for name in experiments:
            group = [e for e in runs if e["experiment"] == name]
            done = sum(os.path.exists(e["file"]) for e in group)
            print(f"{name:20s}{group[0]['paper']:22s}{len(group):>6d}{done:>6d}")
        print(f"\nconfigs: {', '.join(CONFIGS)}\nseeds:   {', '.join(map(str, SEEDS))}")
        return

    todo = selected(args, runs)
    if args.check:
        missing = [e for e in todo if not os.path.exists(e["file"])]
        for e in missing:
            print(f"missing  {e['paper']:22s}{e['config']:16s}{e['script']:22s}seed {e['seed']}")
        print(f"\n{len(todo) - len(missing)}/{len(todo)} runs have a result file.")
        sys.exit(1 if missing else 0)

    if not args.force:
        todo = [e for e in todo if not os.path.exists(e["file"])]
    if not todo:
        print("Nothing to do: every selected run already has a result file (use --force to recompute).")
        return

    print(f"{len(todo)} run(s) to launch.\n")
    for i, entry in enumerate(todo, 1):
        cmd = command(entry)
        print(f"[{i}/{len(todo)}] {entry['config']} {entry['script']} seed {entry['seed']}")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        result = subprocess.run(cmd, cwd=SCRIPTS)
        if result.returncode != 0:
            sys.exit(f"Run failed ({result.returncode}); stopping here.")


if __name__ == "__main__":
    main()
