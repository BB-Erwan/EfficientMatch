"""Post-traitement des logs JSON produits par train.py : calcule les métriques du papier (Table 1)
qui ne sont PAS calculées pendant l'entraînement -- AUC normalisée sur [0, K], report de dernière
valeur EMA pour les runs arrêtés tôt, itérations/FLOPs pour atteindre un seuil de performance défini
par rapport à l'accuracy asymptotique de FixMatch dans NOTRE protocole (cf. PROJECT_SPEC.md §4/§5,
PDF §"Critère d'arrêt anticipé").

Usage :
    python analyze.py --logs-dir ./logs --dataset cifar10 --n-labels 250 --K 131072 \
        --reference-algo fixmatch --threshold-frac 0.9

    # Inspecter l'ablation lambda_mix (Phase 2, K=2**14, tags lammix0.5/lammix1.0/lammix2.0) ou
    # l'étude de sensibilité à mu (tags mu3/mu5/mu7) -- même mécanisme de regroupement par tag :
    python analyze.py --logs-dir ./logs --dataset cifar10 --n-labels 250 --K 16384
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# np.trapz a été retiré en NumPy 2.0+ au profit de np.trapezoid -- on garde la compatibilité
# avec les deux séries (l'environnement d'entraînement peut être figé sur une NumPy 1.x).
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

LOG_NAME_RE = re.compile(
    r"^(?P<algo>[a-z_]+)_(?P<dataset>[a-z0-9]+)_n(?P<n_labels>\d+)_K(?P<K>\d+)_seed(?P<seed>\d+)"
    r"(?:_(?P<tag>[A-Za-z0-9.]+))?\.json$"
)


def load_run(path):
    with open(path) as f:
        data = json.load(f)
    if data.get("status") != "completed":
        raise ValueError(
            f"{path} n'est pas marqué comme complet (status={data.get('status')!r}) -- "
            "run interrompu/en cours, à exclure de l'analyse ou à relancer."
        )
    return data


def curve_from_logs(run):
    logs = run["logs"]
    iterations = np.array([e["iteration"] for e in logs], dtype=float)
    accuracies = np.array([e["eval_accuracy"] for e in logs], dtype=float)
    flops = np.array([e["cumulative_flops"] for e in logs], dtype=float)
    return iterations, accuracies, flops


def forward_fill_to_K(iterations, accuracies, flops, K):
    """Complète la courbe jusqu'à K par report de la dernière valeur observée -- accuracy EMA ET
    FLOPs cumulés (l'entraînement étant arrêté, aucun FLOP supplémentaire n'est dépensé) -- pour que
    l'AUC reste calculable et comparable entre méthodes arrêtées à des itérations différentes
    (cf. papier, §"Critère d'arrêt anticipé")."""
    if len(iterations) == 0 or iterations[-1] >= K:
        return iterations, accuracies, flops
    return (
        np.append(iterations, K),
        np.append(accuracies, accuracies[-1]),
        np.append(flops, flops[-1]),
    )


def compute_auc(iterations, accuracies, K, chance_level):
    """AUC de la courbe accuracy-vs-itérations, normalisée sur [0, K] (papier, Table 1). Le point
    (0, chance_level) est ajouté comme origine : l'accuracy avant toute mise à jour de poids est
    celle d'un classifieur aléatoire (1/num_classes), pas 0."""
    x = np.concatenate([[0.0], iterations])
    y = np.concatenate([[chance_level], accuracies])
    return float(_trapezoid(y, x) / K)


def iters_flops_to_threshold(iterations, accuracies, flops, threshold):
    """Première itération (et FLOPs cumulés correspondants) où l'accuracy atteint `threshold`.
    Retourne (None, None) si jamais atteint, y compris après report de dernière valeur."""
    reached = np.where(accuracies >= threshold)[0]
    if len(reached) == 0:
        return None, None
    idx = int(reached[0])
    return float(iterations[idx]), float(flops[idx])


def summarize_single_run(path, K, threshold=None):
    run = load_run(path)
    cfg = run["config"]
    num_classes = cfg["num_classes"]
    iterations, accuracies, flops = curve_from_logs(run)
    iterations, accuracies, flops = forward_fill_to_K(iterations, accuracies, flops, K)
    auc = compute_auc(iterations, accuracies, K, chance_level=1.0 / num_classes)
    final_acc = float(accuracies[-1]) if len(accuracies) else float("nan")
    iters_thr, flops_thr = (None, None)
    if threshold is not None:
        iters_thr, flops_thr = iters_flops_to_threshold(iterations, accuracies, flops, threshold)
    return {
        "path": str(path), "seed": cfg["seed"], "auc": auc, "final_accuracy": final_acc,
        "iters_to_threshold": iters_thr, "flops_to_threshold": flops_thr,
        "stopped_early": run.get("stopped_early", False), "last_iteration": run.get("last_iteration"),
    }


def discover_runs(logs_dir):
    """Regroupe les logs par (algo, dataset, n_labels, K, tag) -- une entrée par graine.
    Le tag est inclus dans la clé pour ne jamais mélanger des variantes d'ablation (ex. plusieurs
    valeurs de lambda_mix) qui partagent (algo, dataset, n_labels, K, seed)."""
    groups = defaultdict(list)
    for path in sorted(glob.glob(str(Path(logs_dir) / "*.json"))):
        m = LOG_NAME_RE.match(Path(path).name)
        if not m:
            continue
        key = (m["algo"], m["dataset"], int(m["n_labels"]), int(m["K"]), m["tag"] or "")
        groups[key].append(path)
    return groups


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def format_pair(pair, fmt):
    mean, std = pair
    if mean is None:
        return "-"
    return f"{mean:{fmt}}±{std:{fmt}}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", default="./logs")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--n-labels", type=int, default=250)
    parser.add_argument("--K", type=int, default=2 ** 17)
    parser.add_argument("--reference-algo", default="fixmatch",
                         help="algo dont l'accuracy asymptotique définit le seuil de performance")
    parser.add_argument("--threshold-frac", type=float, default=0.9,
                         help="fraction de l'accuracy asymptotique de --reference-algo -- placeholder "
                              "papier (PROJECT_SPEC.md §6 : valeur non encore tranchée, 0.9 = exemple)")
    parser.add_argument("--out", default=None, help="chemin JSON optionnel pour écrire le tableau récapitulatif")
    args = parser.parse_args()

    groups = discover_runs(args.logs_dir)
    relevant = {
        k: v for k, v in groups.items()
        if k[1] == args.dataset and k[2] == args.n_labels and k[3] == args.K
    }
    if not relevant:
        print(f"Aucun log complet trouvé dans {args.logs_dir} pour dataset={args.dataset}, "
              f"n_labels={args.n_labels}, K={args.K}.")
        return

    ref_paths = [v for k, v in relevant.items() if k[0] == args.reference_algo and k[4] == ""]
    reference_accuracy = None
    if ref_paths:
        ref_runs = [summarize_single_run(p, args.K) for p in ref_paths[0]]
        reference_accuracy = float(np.mean([r["final_accuracy"] for r in ref_runs]))
        print(f"Accuracy asymptotique de référence ({args.reference_algo}) : {reference_accuracy:.4f} "
              f"(moyenne sur {len(ref_runs)} graine(s))")
    else:
        print(f"ATTENTION : pas de run complet pour l'algo de référence '{args.reference_algo}' -- "
              "la métrique itérations/FLOPs jusqu'à seuil ne sera pas calculée.")

    threshold = reference_accuracy * args.threshold_frac if reference_accuracy is not None else None

    table = {}
    for (algo, dataset, n_labels, K, tag), paths in sorted(relevant.items()):
        label = f"{algo}[{tag}]" if tag else algo
        summaries = [summarize_single_run(p, K, threshold) for p in paths]
        table[label] = {
            "n_seeds": len(summaries),
            "auc": mean_std([s["auc"] for s in summaries]),
            "final_accuracy": mean_std([s["final_accuracy"] for s in summaries]),
            "iters_to_threshold": mean_std([s["iters_to_threshold"] for s in summaries]),
            "flops_to_threshold": mean_std([s["flops_to_threshold"] for s in summaries]),
            "n_stopped_early": sum(1 for s in summaries if s["stopped_early"]),
        }

    header = f"{'Méthode':<20}{'AUC':>18}{'Acc. finale':>18}{'Itérations':>20}{'FLOPs':>18}{'n':>4}"
    print("\n" + header)
    print("-" * len(header))
    for label, row in table.items():
        print(
            f"{label:<20}"
            f"{format_pair(row['auc'], '.4f'):>18}"
            f"{format_pair(row['final_accuracy'], '.4f'):>18}"
            f"{format_pair(row['iters_to_threshold'], '.0f'):>20}"
            f"{format_pair(row['flops_to_threshold'], '.2e'):>18}"
            f"{row['n_seeds']:>4}"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "reference_accuracy": reference_accuracy, "threshold": threshold,
                "threshold_frac": args.threshold_frac, "table": table,
            }, f, indent=2)
        print(f"\nRésumé écrit dans {args.out}")


if __name__ == "__main__":
    main()
