"""Regenerates BEST_METHOD_BY_THRESHOLD.md from results/ (5 main methods, seuils target-5% / target-10%)."""
import json, os
from collections import Counter
from run_analysis import EVAL_SECONDS, first_reaching, corrected_minutes, gflops_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = (2312, 308, 2701)
METHODS = ["efficientmatch_3", "fixmatch", "flexmatch", "mixmatch", "regmixmatch"]
CONFIGS = [  # (label, dataset, n_labeled, target, widen_factor, file suffix)
    ("SVHN, 250 labels", "svhn", 250, 0.90, 2, ""),
    ("CIFAR-10, 250 labels", "cifar10", 250, 0.80, 2, ""),
    ("CIFAR-10, 4000 labels", "cifar10", 4000, 0.90, 2, ""),
    ("CIFAR-100, 2500 labels (WF4)", "cifar100", 2500, 0.50, 4, "_wf4"),
]


def measure(ds, n, seed, m, thr, wf, suf):
    p = os.path.join(ROOT, "results", f"{ds}-labeled-{n}-seed-{seed}", f"{m}_ema{suf}_metrics.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    hit = first_reaching(d, thr)
    if hit is None:
        return None
    i, step, _, t = hit
    if t / 60 > 120:
        return None
    return corrected_minutes(t, i + 1, EVAL_SECONDS[wf]), gflops_for(m + "_ema", wf, ds) * step / 1000


def winner(vals):
    vals = {k: v for k, v in vals.items() if v is not None}
    return min(vals, key=vals.get) if vals else "aucune"


def majority(ws):
    c = Counter(ws).most_common()
    if c[0][1] >= 2 and (len(c) == 1 or c[1][1] < c[0][1]):
        return c[0][0]
    return "pas de majorité"


detail, summary = {}, {}
for label, ds, n, target, wf, suf in CONFIGS:
    for red in (0.05, 0.10):
        thr = round(target - red, 2)
        wt, wfl, rows = [], [], []
        for seed in SEEDS:
            res = {m: measure(ds, n, seed, m, thr, wf, suf) for m in METHODS}
            bt = winner({m: (v[0] if v else None) for m, v in res.items()})
            bf = winner({m: (v[1] if v else None) for m, v in res.items()})
            wt.append(bt); wfl.append(bf)
            rows.append((thr, seed, bt, bf, res))
        detail[(label, red)] = rows
        summary[(label, red)] = (majority(wt), majority(wfl))

out = []
w = out.append
w("# Meilleure méthode par seuil d'accuracy réduit (-5% / -10%)\n")
w("Ce document (généré par `scripts/best_method_by_threshold.py`) identifie, pour chacune des 4 configurations "
  "retenues de `EXPERIENCE_PRINCIPALE.md` (CIFAR-100 10000 labels exclu), quelle méthode parmi les 5 méthodes "
  "principales (efficientmatch_3, fixmatch, flexmatch, mixmatch, regmixmatch) atteint le plus vite (temps corrigé "
  "du coût des évaluations) et au moindre coût (FLOPs cumulés) un seuil d'accuracy réduit de 5 ou 10 points par "
  "rapport au target_acc habituel de la configuration. La variante efficientmatch_freematch n'est pas incluse.\n")
w("Seuils (target original → -5% → -10%) :\n")
w("| Configuration | Target original | Seuil -5% | Seuil -10% |\n|---|---:|---:|---:|")
for label, ds, n, target, wf, suf in CONFIGS:
    w(f"| {label} | {target*100:.0f}% | {(target-.05)*100:.0f}% | {(target-.10)*100:.0f}% |")
w("\n## Tableau récapitulatif (méthode gagnante, majorité sur 3 seeds)\n")
w("| Configuration | -5% (temps) | -5% (FLOPs) | -10% (temps) | -10% (FLOPs) |\n|---|---|---|---|---|")
for label, *_ in CONFIGS:
    a, b = summary[(label, 0.05)], summary[(label, 0.10)]
    w(f"| {label} | {a[0]} | {a[1]} | {b[0]} | {b[1]} |")
w("\n## Détail par seed\n")
for label, ds, n, target, wf, suf in CONFIGS:
    w(f"### {label}\n")
    w("| Seuil | Seed | Meilleur temps | Meilleur FLOPs | Temps corrigé (min) par méthode | TFLOPs par méthode |\n|---|---|---|---|---|---|")
    for red in (0.05, 0.10):
        for thr, seed, bt, bf, res in detail[(label, red)]:
            tm = ", ".join(f"{m} {v[0]:.1f}" if v else f"{m} n/a" for m, v in res.items())
            fl = ", ".join(f"{m} {v[1]:,.0f}".replace(",", " ") if v else f"{m} n/a" for m, v in res.items())
            w(f"| -{red*100:.0f}% ({thr*100:.0f}%) | {seed} | {bt} | {bf} | {tm} | {fl} |")
    w("")
w("`n/a` : seuil jamais atteint par la run.\n")
w("## Reproduire\n\n```bash\npython scripts/best_method_by_threshold.py\n```\n")
open(os.path.join(ROOT, "BEST_METHOD_BY_THRESHOLD.md"), "w", encoding="utf-8").write("\n".join(out))
