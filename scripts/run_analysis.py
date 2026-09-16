"""Reusable analysis tool for the questions asked repeatedly about a run/config during this
project: full comparison table, comparison "at equivalent accuracy" against an in-progress run,
and "corrected time" (raw time minus evaluation overhead). Reads metrics JSON files directly --
no training code involved, safe to run alongside an active run.

Eval-time-per-call and FLOPs/iteration constants below come from `FLOPS_RESULTS.md` (in-situ
measurements via scripts/ghost_method.py, and scripts/flops_analysis.py) -- update both places if
either changes.

Widen factor (architecture) is detected PER FILE, not per directory: a training script now
appends "_wf{N}" to its output filename whenever --widen_factor differs from its own default (2),
so a single results directory (typically a CIFAR-100 config) can legitimately mix WRN-28-2/4/8
runs -- e.g. `fixmatch_ema_metrics.json` (implicit default, historically WF8 for the early
CIFAR-100 runs made before this suffix existed) next to `fixmatch_ema_wf4_metrics.json`. Use
--default_widen_factor to override what a bare, unsuffixed filename should be assumed to mean for
older files (defaults to 8 for cifar100, 2 otherwise).

Usage:
    # Full comparison table for a config, sorted by best accuracy, includes corrected time.
    python run_analysis.py compare --dataset cifar10 --num_labeled 250 --seed 2312
    python run_analysis.py compare --dir ../results/svhn-labeled-250-seed-2312

    # "At equivalent accuracy": for every other method in the config, find the step/time at which
    # it first reached the same accuracy as --like's current (or best) accuracy.
    python run_analysis.py at-acc --dataset cifar100 --num_labeled 10000 --seed 2312 --like efficientmatch_3_ema_wf4
    python run_analysis.py at-acc --dataset cifar10 --num_labeled 250 --seed 2312 --target 0.75

    # Just the current status of one method's run (step/acc/time, raw and corrected).
    python run_analysis.py status --dataset cifar10 --num_labeled 250 --seed 2312 --method efficientmatch_3
"""
import argparse
import glob
import json
import os
import re

# GFLOPs per training iteration, keyed by widen_factor then by method name. See FLOPS_RESULTS.md.
GFLOPS_PER_IT = {
    2: {
        "fixmatch": 850.10, "flexmatch": 850.10, "mixmatch": 301.64,
        "efficientmatch": 740.35, "efficientmatch_2": 740.35, "efficientmatch_3": 740.35,
        "efficientmatch_3_mu1": 356.46, "efficientmatch_3_mu5": 1124.25, "efficientmatch_3_mu7": 1508.14,
        "efficientmatch_flex": 740.35, "efficientmatch_flex_mu2": 548.40, "efficientmatch_2_flex": 740.35,
        "sequencematch": 1809.61, "regmixmatch": 1891.86, "regmixmatch_mu3": 904.80,
    },
    4: {
        "fixmatch": 3354.88, "flexmatch": 3354.88, "mixmatch": 1190.43,
        "regmixmatch": 7467.01, "efficientmatch_3": 2921.93,
    },
    8: {
        "fixmatch": 13332.36, "flexmatch": 13332.36, "mixmatch": 4730.83,
    },
}

# Mean wall-clock cost of one evaluate_f1_and_accuracy() call (EMA copy included), measured in
# situ via scripts/ghost_method.py, per widen_factor.
EVAL_SECONDS = {2: 0.5537, 4: 1.4991, 8: 4.4763}

WF_SUFFIX_RE = re.compile(r"_wf(\d+)$")


def default_widen_factor(args, result_dir):
    if args.default_widen_factor:
        return args.default_widen_factor
    return 8 if "cifar100" in result_dir else 2


def parse_widen_factor(name, default_wf):
    """Returns (base_name_without_wf_suffix, widen_factor) for a method name as it appears in a
    filename (e.g. 'fixmatch_ema_wf4' -> ('fixmatch_ema', 4); 'fixmatch_ema' -> ('fixmatch_ema', default_wf))."""
    m = WF_SUFFIX_RE.search(name)
    if m:
        return name[:m.start()], int(m.group(1))
    return name, default_wf


def gflops_for(name, widen_factor):
    table = GFLOPS_PER_IT.get(widen_factor, {})
    n = name.replace("_ema", "")
    if n in table:
        return table[n]
    parts = n.split("_")
    for i in range(len(parts), 0, -1):
        cand = "_".join(parts[:i])
        if cand in table:
            return table[cand]
    return None


def resolve_dir(args):
    if args.dir:
        return args.dir
    if not (args.dataset and args.num_labeled and args.seed):
        raise SystemExit("Pass either --dir, or all of --dataset/--num_labeled/--seed")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                         f"{args.dataset}-labeled-{args.num_labeled}-seed-{args.seed}")


def load_runs(result_dir, default_wf):
    """Returns {display_name: (data, widen_factor)}. display_name is the filename's method name
    as-is (including any _wfN suffix), so distinct architectures never collide."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(result_dir, "*_metrics.json"))):
        name = os.path.basename(f).replace("_metrics.json", "")
        with open(f) as fh:
            data = json.load(fh)
        if data.get("test_acc"):
            _, wf = parse_widen_factor(name, default_wf)
            runs[name] = (data, wf)
    return runs


def best_point(data):
    """(index, step, acc, time_s) of the best accuracy reached so far."""
    acc = data["test_acc"]
    idx = acc.index(max(acc))
    return idx, data["step"][idx], acc[idx], data["time_elapsed"][idx]


def first_reaching(data, target):
    """(index, step, acc, time_s) of the first evaluation reaching >= target, or None."""
    for i, a in enumerate(data["test_acc"]):
        if a >= target:
            return i, data["step"][i], a, data["time_elapsed"][i]
    return None


def corrected_minutes(time_s, n_evals, eval_s):
    return (time_s - n_evals * eval_s) / 60


def cmd_compare(args):
    result_dir = resolve_dir(args)
    default_wf = default_widen_factor(args, result_dir)
    runs = load_runs(result_dir, default_wf)
    if not runs:
        raise SystemExit(f"No *_metrics.json found in {result_dir}")

    rows = []
    for name, (data, wf) in runs.items():
        eval_s = EVAL_SECONDS.get(wf)
        idx, step, acc, time_s = best_point(data)
        n_evals = idx + 1
        gflops_it = gflops_for(name, wf)
        tflops = gflops_it * step / 1000 if gflops_it else None
        corr = corrected_minutes(time_s, n_evals, eval_s) if eval_s else None
        rows.append((name, wf, acc, step, time_s / 60, corr, tflops, n_evals))
    rows.sort(key=lambda r: -r[2])

    print(f"\n{result_dir}  (default WF for unsuffixed files: {default_wf})\n")
    header = f"{'method':<28}{'wf':>4}{'best_acc':>10}{'@step':>10}{'@min':>8}{'corrected':>11}{'TFLOPs':>11}{'evals':>7}"
    print(header)
    print("-" * len(header))
    for name, wf, acc, step, tm, corr, tflops, n in rows:
        tflops_str = f"{tflops:,.0f}".replace(",", " ") if tflops else "N/A"
        corr_str = f"{corr:.1f}m" if corr is not None else "N/A"
        print(f"{name:<28}{wf:>4}{acc*100:>9.2f}%{step:>10}{tm:>7.1f}m{corr_str:>11}{tflops_str:>11}{n:>7}")


def cmd_status(args):
    result_dir = resolve_dir(args)
    default_wf = default_widen_factor(args, result_dir)
    runs = load_runs(result_dir, default_wf)
    if args.method not in runs:
        raise SystemExit(f"No metrics file for method '{args.method}' in {result_dir}. Available: {sorted(runs)}")
    data, wf = runs[args.method]
    eval_s = EVAL_SECONDS.get(wf)
    idx, step, acc, time_s = best_point(data)
    n_evals = idx + 1
    corr_str = f"{corrected_minutes(time_s, n_evals, eval_s):.2f}min" if eval_s else "N/A"
    print(f"{args.method} (WRN-28-{wf}): best step={step} acc={acc*100:.2f}% time={time_s/60:.2f}min "
          f"corrected={corr_str} n_evals={len(data['test_acc'])} "
          f"(last logged step={data['step'][-1]}, acc={data['test_acc'][-1]*100:.2f}%)")

    if args.tail:
        print(f"\nLast {args.tail} evaluations:")
        for i in range(max(0, len(data["test_acc"]) - args.tail), len(data["test_acc"])):
            print(f"  step={data['step'][i]:<8} acc={data['test_acc'][i]*100:>6.2f}%  t={data['time_elapsed'][i]/60:>7.2f}min")


def cmd_at_acc(args):
    result_dir = resolve_dir(args)
    default_wf = default_widen_factor(args, result_dir)
    runs = load_runs(result_dir, default_wf)

    if args.target is not None:
        target = args.target
        ref_name = None
    elif args.like:
        if args.like not in runs:
            raise SystemExit(f"No metrics file for method '{args.like}' in {result_dir}. Available: {sorted(runs)}")
        ref_data, ref_wf = runs[args.like]
        _, ref_step, target, ref_time_s = best_point(ref_data)
        ref_name = args.like
        print(f"Reference: {ref_name} (WRN-28-{ref_wf}) currently at acc={target*100:.2f}% "
              f"(step={ref_step}, t={ref_time_s/60:.2f}min)\n")
    else:
        raise SystemExit("Pass either --target <accuracy 0-1> or --like <method name>")

    rows = []
    for name, (data, wf) in runs.items():
        if name == ref_name:
            continue
        eval_s = EVAL_SECONDS.get(wf)
        hit = first_reaching(data, target)
        if hit is None:
            rows.append((name, wf, None, None, None, max(data["test_acc"])))
        else:
            idx, step, acc, time_s = hit
            n_evals = idx + 1
            corr = corrected_minutes(time_s, n_evals, eval_s) if eval_s else None
            rows.append((name, wf, step, time_s / 60, corr, acc))

    rows.sort(key=lambda r: (r[2] is None, r[3] if r[3] is not None else 0))
    print(f"At accuracy >= {target*100:.2f}%:\n")
    for name, wf, step, tm, corr, acc in rows:
        if tm is None:
            print(f"  {name:<28} (WRN-28-{wf}) never reached (max {acc*100:.2f}%)")
        else:
            corr_str = f"{corr:.2f}min" if corr is not None else "N/A"
            print(f"  {name:<28} (WRN-28-{wf}) step={step:<8} t={tm:>7.2f}min  corrected={corr_str:>9}  acc={acc*100:.2f}%")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", type=str)
    common.add_argument("--num_labeled", type=int)
    common.add_argument("--seed", type=int)
    common.add_argument("--dir", type=str, default=None, help="Results directory directly (overrides --dataset/--num_labeled/--seed)")
    common.add_argument("--default_widen_factor", type=int, default=None,
                         help="Architecture assumed for files with no _wfN suffix (default: cifar100 -> 8, else -> 2). "
                              "Files with an explicit _wfN suffix always use that value regardless of this flag.")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compare = sub.add_parser("compare", help="Full comparison table for a config", parents=[common])
    p_compare.set_defaults(func=cmd_compare)

    p_status = sub.add_parser("status", help="Current status of one method's run", parents=[common])
    p_status.add_argument("--method", type=str, required=True, help="e.g. efficientmatch_3_ema, efficientmatch_3_ema_wf4")
    p_status.add_argument("--tail", type=int, default=0, help="Also print the last N evaluations (step/acc/time), useful to see if a no-target_acc run has plateaued")
    p_status.set_defaults(func=cmd_status)

    p_at_acc = sub.add_parser("at-acc", help="Compare methods at equivalent accuracy", parents=[common])
    group = p_at_acc.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", type=float, help="Target accuracy as a 0-1 fraction")
    group.add_argument("--like", type=str, help="Use this method's current best accuracy as the target")
    p_at_acc.set_defaults(func=cmd_at_acc)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
