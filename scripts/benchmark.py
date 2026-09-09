"""Mesure, pour chaque algorithme SSL de ce dépôt, le temps réel par itération et les FLOPs par
itération sur la machine cible -- pas de run complet, juste de quoi estimer le temps total avant
d'en lancer un. Écrit un tableau récapitulatif à l'écran et, avec --out, un document Markdown.

Réutilise `estimate_flops_per_iter(args, device)` et `estimate_time_per_iter(args, device, ...)`,
définies dans chaque script d'algorithme (fixmatch.py, flexmatch.py, mixmatch.py, efficientmatch.py)
-- ce script se contente d'orchestrer les 4 mesures et de les mettre en forme, il ne réimplémente
aucune logique d'algorithme.

Usage :
    python benchmark.py
    python benchmark.py --algos fixmatch efficientmatch --bench-iters 50
    python benchmark.py --K 131072 --n-labels 250 --out ../BENCHMARK_RESULTS.md
"""
import argparse
import platform
import time
from datetime import date
from pathlib import Path

import torch

import efficientmatch
import fixmatch
import flexmatch
import mixmatch

MODULES = {
    "fixmatch": fixmatch,
    "flexmatch": flexmatch,
    "mixmatch": mixmatch,
    "efficientmatch": efficientmatch,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algos", nargs="+", default=sorted(MODULES), choices=sorted(MODULES))
    parser.add_argument("--n-labels", type=int, default=250)
    parser.add_argument("--K", type=int, default=2 ** 17, help="budget utilisé pour l'extrapolation du temps total")
    parser.add_argument("--warmup-iters", type=int, default=20,
                         help="itérations de chauffe non chronométrées (stabilise cudnn.benchmark/"
                              "torch.compile/l'allocateur CUDA)")
    parser.add_argument("--bench-iters", type=int, default=100,
                         help="itérations chronométrées et moyennées -- volontairement pas trop petit : "
                              "le tout début de l'entraînement peut être irrégulier (recompilations "
                              "tardives, effets de cache), un nombre trop faible surestimerait la "
                              "précision de la moyenne mesurée")
    parser.add_argument("--debug-subset-size", type=int, default=4096,
                         help="sous-ensemble non labellisé utilisé pour la mesure (pas pour les runs réels)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                         help="mesure avec torch.compile (comme un run réel) -- ajoute un coût de "
                              "compilation ponctuel par algo, absorbé par la chauffe si --warmup-iters "
                              "est assez grand")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0,
                         help="0 par défaut : évite le coût/risque de démarrage des workers "
                              "multiprocessing pour une mesure de quelques secondes -- augmenter pour "
                              "mesurer l'effet réel du parallélisme des workers")
    parser.add_argument("--flops-only", action="store_true",
                         help="ne mesure que les FLOPs par itération (rapide, pas de CIFAR-10 ni de "
                              "chronométrage réel) -- saute entièrement estimate_time_per_iter")
    parser.add_argument("--out", type=str, default=None, help="chemin du document Markdown récapitulatif")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if gpu_name:
        print(f"GPU: {gpu_name}")

    results = {}
    for name in args.algos:
        module = MODULES[name]
        print(f"\n[{name}]")
        overrides = [
            "--n-labels", str(args.n_labels), "--K", str(args.K),
            "--debug-subset-size", str(args.debug_subset_size),
            "--num-workers", str(args.num_workers),
        ]
        overrides += ["--compile"] if args.compile else ["--no-compile"]
        overrides += ["--amp"] if args.amp else ["--no-amp"]
        overrides += ["--pin-memory"] if args.pin_memory else ["--no-pin-memory"]
        algo_args = module.parse_args(overrides)

        print("    FLOPs par itération (modèle jetable, données factices)...", flush=True)
        flops = module.estimate_flops_per_iter(algo_args, device)
        results[name] = {"flops_per_iter": flops}
        print(f"    -> {flops:.3e} FLOPs/it")

        if args.flops_only:
            continue

        print(f"    temps par itération ({args.warmup_iters} chauffe + {args.bench_iters} mesurées, "
              f"CIFAR-10 réel, {args.debug_subset_size} images non labellisées)...", flush=True)
        t0 = time.time()
        mean_s, std_s = module.estimate_time_per_iter(algo_args, device, args.warmup_iters, args.bench_iters)
        measure_wall_time = time.time() - t0

        it_per_sec = 1.0 / mean_s
        total_hours = args.K * mean_s / 3600
        # coefficient de variation (écart-type / moyenne) : repère une mesure encore instable
        # (ex. recompilation tardive en cours de mesure) indépendamment de l'échelle absolue du temps.
        cv_pct = 100 * std_s / mean_s
        results[name].update({
            "seconds_per_iter": mean_s, "seconds_per_iter_std": std_s,
            "it_per_sec": it_per_sec, "total_hours_at_K": total_hours,
            "measure_wall_time": measure_wall_time,
        })
        print(f"    -> {mean_s * 1000:.1f}±{std_s * 1000:.1f} ms/it (CV={cv_pct:.1f}%), "
              f"{it_per_sec:.2f} it/s, {total_hours:.2f} h pour K={args.K} (mesure : {measure_wall_time:.0f}s)")
        if cv_pct > 15:
            print(f"    ATTENTION : écart-type élevé ({cv_pct:.0f}% de la moyenne) -- mesure possiblement "
                  f"encore instable, envisager --warmup-iters/--bench-iters plus grands pour {name}.")

    has_timing = not args.flops_only
    if has_timing:
        header = f"{'Algo':<16}{'FLOPs/it':>14}{'ms/it':>14}{'it/s':>8}{f'Temps total (K={args.K})':>26}"
    else:
        header = f"{'Algo':<16}{'FLOPs/it':>14}"
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        if has_timing:
            ms_col = f"{r['seconds_per_iter'] * 1000:.1f}±{r['seconds_per_iter_std'] * 1000:.1f}"
            print(f"{name:<16}{r['flops_per_iter']:>14.3e}{ms_col:>14}{r['it_per_sec']:>8.2f}"
                  f"{r['total_hours_at_K']:>23.2f} h")
        else:
            print(f"{name:<16}{r['flops_per_iter']:>14.3e}")

    total_seq = sum(r["total_hours_at_K"] for r in results.values()) if has_timing else None
    if has_timing:
        print(f"\nTotal séquentiel pour les {len(results)} algo(s) sélectionné(s) (1 graine, K={args.K}) : "
              f"{total_seq:.1f} h")

    if args.out:
        write_markdown(args.out, args, device, gpu_name, results, total_seq)
        print(f"\nRésumé écrit dans {args.out}")


def write_markdown(out_path, args, device, gpu_name, results, total_seq):
    has_timing = total_seq is not None
    title = "Résultats du benchmark de FLOPs" + (" et de vitesse" if has_timing else "")
    lines = [
        f"# {title}",
        "",
        f"Mesures réalisées via `scripts/benchmark.py` le {date.today().isoformat()}, sur "
        f"{gpu_name or device.type} (`torch=={torch.__version__}`, Python {platform.python_version()}).",
        "",
        "## Configuration de la mesure",
        "",
        "| Paramètre | Valeur |",
        "|---|---|",
        "| Dataset | CIFAR-10 |",
        f"| n_labels | {args.n_labels} |",
    ]
    if has_timing:
        lines += [
            f"| Sous-ensemble non labellisé (mesure uniquement) | {args.debug_subset_size} |",
            f"| Budget cible K (pour l'extrapolation) | {args.K} |",
            f"| Itérations de chauffe / mesurées | {args.warmup_iters} / {args.bench_iters} |",
            f"| `torch.compile` | {'activé (mode par défaut, pas `reduce-overhead`)' if args.compile else 'désactivé'} |",
            f"| `torch.autocast` (bfloat16) | {'activé' if args.amp else 'désactivé'} |",
            f"| `pin_memory` / `non_blocking` | {'activé' if args.pin_memory else 'désactivé'} |",
            f"| `num_workers` (DataLoader) | {args.num_workers} |",
        ]
    lines += [
        "",
        "Les FLOPs sont mesurés une seule fois par algo (`torch.utils.flop_counter.FlopCounterMode`, "
        "forward + backward) sur un modèle jetable et des tenseurs factices de la bonne forme -- "
        "indépendants des vraies données, donc indépendants aussi de `--compile`/`--amp`.",
    ]
    if has_timing:
        lines.append(
            "Le temps par itération est chronométré INDIVIDUELLEMENT sur chacune des itérations "
            "mesurées (boucle réelle, mêmes appels que `main()`, CIFAR-10 réel, sans EMA ni "
            "logging/évaluation) plutôt qu'en moyenne globale sur tout le bloc -- l'écart-type reporté "
            "permet de voir si la mesure est stable (le tout début de l'entraînement peut être "
            "irrégulier : recompilation tardive d'une branche pas vue pendant la chauffe, effets de "
            "cache)."
        )
    lines += ["", "## Résultats", ""]
    if has_timing:
        lines += [
            "| Méthode | FLOPs / itération | ms / itération | it/s | Temps total estimé (K={}) |".format(args.K),
            "|---|---:|---:|---:|---:|",
        ]
        for name, r in results.items():
            ms_col = f"{r['seconds_per_iter'] * 1000:.1f} ± {r['seconds_per_iter_std'] * 1000:.1f}"
            lines.append(
                f"| {name} | {r['flops_per_iter']:.3e} | {ms_col} | {r['it_per_sec']:.2f} | "
                f"{r['total_hours_at_K']:.2f} h |"
            )
        lines += [
            "",
            f"Total séquentiel pour les {len(results)} algorithme(s) mesuré(s) ci-dessus (1 graine, "
            f"K={args.K}) : **{total_seq:.1f} h**.",
        ]
    else:
        lines += ["| Méthode | FLOPs / itération |", "|---|---:|"]
        for name, r in results.items():
            lines.append(f"| {name} | {r['flops_per_iter']:.3e} |")
    lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
