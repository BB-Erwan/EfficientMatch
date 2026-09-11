"""Trace test_acc en fonction du temps pour des runs seed 2312 (RTX 5060 Ti, EMA), avec en plus
une courbe "temps équivalent RTX A4000" estimée à partir du facteur s/itération par méthode
calculé par gpu_time_equivalence.py (cf. ce script pour la méthodologie).

Le facteur est mesuré sur des runs SANS EMA (seule donnée A4000 disponible) et appliqué tel
quel aux runs EMA de la seed 2312 -- approximation qui suppose que le surcoût de l'EMA
(~+6.5% mesuré séparément) ne dépend pas du GPU. Marqué explicitement dans les titres/légendes.

Usage :
    python scripts/plot_gpu_equivalent_acc_vs_time.py \\
        --gpu-dir "RTX 5060 Ti" results/labeled-250-seed-666 \\
        --gpu-dir "RTX A4000" results/labeled-250-seed-42 \\
        --gpu-dir "RTX A4000" results/labeled-250-seed-0 \\
        --target-dir results/labeled-250-seed-2312 \\
        --reference-gpu "RTX 5060 Ti" \\
        --equivalent-gpu "RTX A4000"
"""
import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gpu_time_equivalence import aggregate, base_method_name, collect  # noqa: E402


def load_curve(metrics_path, test_period):
    with open(metrics_path) as f:
        metrics = json.load(f)
    time_elapsed = metrics["time_elapsed"]
    test_acc = metrics["test_acc"]
    steps = [0] + [i * test_period for i in range(1, len(time_elapsed))]
    return steps, time_elapsed, test_acc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-dir", nargs=2, action="append", metavar=("GPU_NAME", "DIR"), required=True,
                         help="Dossiers de résultats servant à calibrer le facteur s/itération par GPU. Répétable.")
    parser.add_argument("--target-dir", type=str, required=True, help="Dossier des runs à tracer (ex: results/labeled-250-seed-2312).")
    parser.add_argument("--reference-gpu", type=str, required=True, help="GPU sur lequel les runs de --target-dir ont réellement tourné.")
    parser.add_argument("--equivalent-gpu", type=str, required=True, help="GPU pour lequel estimer un temps équivalent.")
    parser.add_argument("--test_period", type=int, default=500)
    parser.add_argument("--calibration-ema-filter", choices=["ema", "non-ema", "all"], default="non-ema",
                         help="Quels runs utiliser pour calibrer le facteur s/itération (défaut: non-ema, seule donnée dispo côté A4000).")
    parser.add_argument("--output_dir", type=str, default=None, help="Défaut : <target-dir>/plots_gpu_equivalent")
    args = parser.parse_args()

    ema_filter = {"ema": True, "non-ema": False, "all": None}[args.calibration_ema_filter]
    per_run = collect(args.gpu_dir, args.test_period, ema_filter=ema_filter)
    agg = aggregate(per_run)

    output_dir = args.output_dir or os.path.join(args.target_dir, "plots_gpu_equivalent")
    os.makedirs(output_dir, exist_ok=True)

    target_files = sorted(glob.glob(os.path.join(args.target_dir, "*_metrics.json")))
    if not target_files:
        raise SystemExit(f"Aucun *_metrics.json trouvé dans {args.target_dir}")

    combined_fig, combined_ax = plt.subplots(figsize=(9, 6))
    skipped = []

    for path in target_files:
        method = base_method_name(path)
        if method is None:
            continue
        label = os.path.basename(path).replace("_metrics.json", "")

        spi_ref = agg.get((args.reference_gpu, method))
        spi_eq = agg.get((args.equivalent_gpu, method))
        if spi_ref is None or spi_eq is None:
            skipped.append(label)
            continue

        steps, time_elapsed, test_acc = load_curve(path, args.test_period)
        eq_time = [s * spi_eq for s in steps]

        # Plot par méthode : temps réel (GPU de référence) vs temps équivalent (GPU cible).
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(time_elapsed, test_acc, label=f"{args.reference_gpu} (réel)", marker=".", markersize=3)
        ax.plot(eq_time, test_acc, label=f"{args.equivalent_gpu} (estimé)", marker=".", markersize=3, linestyle="--")
        ax.set_xlabel("temps (s)")
        ax.set_ylabel("test_acc")
        ax.set_title(f"{label} -- {args.reference_gpu} vs {args.equivalent_gpu} (estimé)")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{label}_acc_vs_time_gpu_equivalent.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

        # Courbe combinée : toutes les méthodes, solide = temps réel, pointillé = équivalent.
        line, = combined_ax.plot(time_elapsed, test_acc, label=f"{label} ({args.reference_gpu})")
        combined_ax.plot(eq_time, test_acc, linestyle="--", color=line.get_color(),
                          label=f"{label} ({args.equivalent_gpu}, estimé)")

    combined_ax.set_xlabel("temps (s)")
    combined_ax.set_ylabel("test_acc")
    combined_ax.set_title(f"seed 2312 -- {args.reference_gpu} (réel, trait plein) vs {args.equivalent_gpu} (estimé, pointillé)")
    combined_ax.legend(fontsize=8)
    combined_fig.tight_layout()
    combined_path = os.path.join(output_dir, "comparison_acc_vs_time_gpu_equivalent.png")
    combined_fig.savefig(combined_path, dpi=150)
    plt.close(combined_fig)
    print(f"Saved {combined_path}")

    if skipped:
        print(f"\nIgnorés (pas de facteur s/itération disponible pour {args.reference_gpu} et/ou {args.equivalent_gpu}) : {', '.join(skipped)}")


if __name__ == "__main__":
    main()
