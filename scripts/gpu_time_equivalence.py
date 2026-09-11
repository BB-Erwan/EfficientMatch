"""Estime un facteur d'équivalence de temps de calcul entre GPU (ex: RTX 5060 Ti avec
torch.compile vs RTX A4000 sans compile -- cf. la condition `"5060" in
torch.cuda.get_device_name(0)` dans fixmatch.py/flexmatch.py/efficientmatch.py/mixmatch.py)
pour chaque méthode, à partir des logs/*_metrics.json déjà produits.

Principe : pour chaque run, on mesure le temps moyen par itération (médiane des intervalles
entre checkpoints `time_elapsed`, en écartant le premier -- qui inclut le chargement du
dataset et le warmup de torch.compile -- et le dernier -- potentiellement plus court en cas
d'arrêt anticipé sur --target_acc). On agrège ce temps/itération par (GPU, méthode), puis :

  temps équivalent sur GPU cible = nb_steps_du_run × temps_par_itération[méthode][GPU cible]

C'est l'approche demandée : pas de mesure FLOPs séparée, juste le temps par itération observé
dans les logs, multiplié par le nombre de steps de l'expérience à convertir.

Usage :
    python scripts/gpu_time_equivalence.py \\
        --gpu-dir "RTX 5060 Ti" results/labeled-250-seed-666 \\
        --gpu-dir "RTX 5060 Ti" results/labeled-250-seed-2312 \\
        --gpu-dir "RTX A4000" results/labeled-250-seed-0 \\
        --gpu-dir "RTX A4000" results/labeled-250-seed-42

    # Convertir un temps précis (ex: temps mesuré sur A4000 -> équivalent 5060 Ti) :
    python scripts/gpu_time_equivalence.py ... \\
        --convert fixmatch "RTX A4000" 14400 "RTX 5060 Ti"
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyze_metrics import FLOPS_PER_ITER  # noqa: E402


def base_method_name(metrics_path):
    """"efficientmatch_flex_ema_metrics.json" -> "efficientmatch" (matches the FLOPS_PER_ITER
    keys, which is also how EMA/adaptive-threshold suffixes get stripped)."""
    name = os.path.basename(metrics_path)
    if name.endswith("_metrics.json"):
        name = name[: -len("_metrics.json")]
    for known in FLOPS_PER_ITER:
        if name.startswith(known):
            return known
    return None


def seconds_per_iter(metrics_path, test_period):
    """Median seconds/iteration for this run, from interior checkpoint intervals only."""
    with open(metrics_path) as f:
        time_elapsed = json.load(f).get("time_elapsed", [])
    if len(time_elapsed) < 4:
        return None, None
    interior_deltas = [time_elapsed[i + 1] - time_elapsed[i] for i in range(1, len(time_elapsed) - 2)]
    if not interior_deltas:
        return None, None
    interior_deltas.sort()
    median_delta = interior_deltas[len(interior_deltas) // 2]
    steps_covered = (len(time_elapsed) - 1) * test_period
    return median_delta / test_period, steps_covered


def collect(gpu_dirs, test_period, ema_filter=None):
    """ema_filter: None (keep everything), True (keep only "*_ema*" files), or False (drop
    "*_ema*" files) -- lets you avoid mixing EMA's own ~6.5% per-iteration overhead into a
    GPU-vs-GPU comparison."""
    per_run = []
    for gpu_name, directory in gpu_dirs:
        for path in sorted(glob.glob(os.path.join(directory, "*_metrics.json"))):
            is_ema = "_ema" in os.path.basename(path)
            if ema_filter is True and not is_ema:
                continue
            if ema_filter is False and is_ema:
                continue
            method = base_method_name(path)
            if method is None:
                continue
            spi, steps_covered = seconds_per_iter(path, test_period)
            if spi is None:
                continue
            per_run.append({
                "gpu": gpu_name,
                "method": method,
                "path": path,
                "seconds_per_iter": spi,
                "steps_covered": steps_covered,
            })
    return per_run


def aggregate(per_run):
    """(gpu, method) -> median seconds/iteration across matching runs (e.g. several seeds)."""
    groups = defaultdict(list)
    for r in per_run:
        groups[(r["gpu"], r["method"])].append(r["seconds_per_iter"])
    agg = {}
    for key, vals in groups.items():
        vals = sorted(vals)
        agg[key] = vals[len(vals) // 2]
    return agg


def print_table(per_run, agg, gpus):
    print(f"{len(per_run)} runs analysés sur {len(gpus)} GPU(s) : {', '.join(gpus)}\n")
    for r in per_run:
        print(f"  [{r['gpu']}] {r['method']:<16} {os.path.basename(r['path']):<40} "
              f"{r['seconds_per_iter']:.4f} s/it  ({r['steps_covered']} steps couverts)")

    print()
    methods = sorted({m for (_, m) in agg})
    header = f"{'méthode':<16}" + "".join(f"{g:>18}" for g in gpus)
    print(header)
    for m in methods:
        row = f"{m:<16}"
        for g in gpus:
            spi = agg.get((g, m))
            row += f"{(f'{spi:.3f} s/it' if spi else 'n/a'):>18}"
        print(row)


def print_factors(agg, gpus, reference):
    print(f"\nFacteur d'équivalence vs {reference!r} (secondes_cible = secondes_source × facteur) :")
    methods = sorted({m for (_, m) in agg})
    for g in gpus:
        if g == reference:
            continue
        print(f"  {g} -> {reference}:")
        for m in methods:
            spi_g = agg.get((g, m))
            spi_ref = agg.get((reference, m))
            if spi_g and spi_ref:
                factor = spi_ref / spi_g
                print(f"    {m:<16} facteur={factor:.3f}  (1s sur {g!r} ≈ {factor:.3f}s sur {reference!r})")
            else:
                print(f"    {m:<16} pas assez de données pour ce couple GPU/méthode")


def convert(agg, method, source_gpu, seconds, target_gpu):
    spi_source = agg.get((source_gpu, method))
    spi_target = agg.get((target_gpu, method))
    if spi_source is None or spi_target is None:
        raise SystemExit(
            f"Pas assez de données pour convertir {method!r} de {source_gpu!r} vers {target_gpu!r} "
            f"(besoin d'au moins un run par GPU pour cette méthode)."
        )
    n_iters = seconds / spi_source
    return n_iters * spi_target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gpu-dir", nargs=2, action="append", metavar=("GPU_NAME", "DIR"), required=True,
        help="Associe un dossier de résultats (results/labeled-...) à un GPU. Répétable (plusieurs dossiers/seeds par GPU).",
    )
    parser.add_argument("--test_period", type=int, default=500, help="test_period utilisé pour les runs (défaut: 500, celui de tous les scripts actuels).")
    parser.add_argument("--ema-only", action="store_true", help="Ne garder que les runs EMA (*_ema*_metrics.json).")
    parser.add_argument("--no-ema-only", action="store_true", help="Ne garder que les runs sans EMA.")
    parser.add_argument("--reference", type=str, default=None, help="GPU de référence pour les facteurs affichés (défaut : le premier --gpu-dir donné).")
    parser.add_argument("--convert", nargs=4, metavar=("METHOD", "SOURCE_GPU", "SECONDS", "TARGET_GPU"), default=None,
                         help="Convertit un temps précis d'un GPU vers un autre pour une méthode donnée.")
    parser.add_argument("--out", type=str, default=None, help="Sauver le tableau agrégé (gpu, méthode, s/it) en CSV.")
    args = parser.parse_args()

    if args.ema_only and args.no_ema_only:
        parser.error("--ema-only et --no-ema-only sont mutuellement exclusifs")
    ema_filter = True if args.ema_only else (False if args.no_ema_only else None)

    per_run = collect(args.gpu_dir, args.test_period, ema_filter=ema_filter)
    if not per_run:
        raise SystemExit("Aucun run exploitable trouvé (vérifie les chemins et --test_period).")

    agg = aggregate(per_run)
    gpus = list(dict.fromkeys(g for g, _ in args.gpu_dir))
    reference = args.reference or gpus[0]

    print_table(per_run, agg, gpus)
    print_factors(agg, gpus, reference)

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gpu", "method", "seconds_per_iter"])
            for (g, m), spi in sorted(agg.items()):
                w.writerow([g, m, spi])
        print(f"\nSaved {args.out}")

    if args.convert:
        method, source_gpu, seconds, target_gpu = args.convert
        seconds = float(seconds)
        eq = convert(agg, method, source_gpu, seconds, target_gpu)
        print(f"\n{seconds:.1f}s sur {source_gpu!r} pour {method!r} ≈ {eq:.1f}s ({eq / 3600:.2f}h) sur {target_gpu!r}")


if __name__ == "__main__":
    main()
