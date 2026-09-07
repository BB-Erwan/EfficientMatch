"""Orchestrateur séquentiel des expériences prioritaires (PROJECT_SPEC.md §7) : lance Phase 2
(ablation lambda_mix) puis Phase 3 (comparaison principale RQ1+RQ2) via subprocess sur train.py,
sur une seule A4000, en série. Reprise automatique : un run dont le fichier de log JSON existe déjà
avec `status == "completed"` est sauté plutôt que relancé.

Exemples :
    # Phase 2 seule (ablation lambda_mix, budget réduit, 1 seed) :
    python run_priority_experiments.py --skip-phase3

    # Phase 3 seule, une fois lambda_mix figé (ex. 1.0 retenu après lecture de analyze.py) :
    python run_priority_experiments.py --skip-phase2 --lambda-mix-frozen 1.0 --phase3-seeds 0 1 2

    # Aperçu de la file sans rien lancer :
    python run_priority_experiments.py --lambda-mix-frozen 1.0 --dry-run
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import build_config  # noqa: E402

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def log_is_complete(log_path):
    path = Path(log_path)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("status") == "completed"


def overrides_to_cli(overrides):
    args = []
    plain_overrides = {}
    for key, value in overrides.items():
        if key in ("dataset", "n_labels", "seed", "K", "tag"):
            flag = "--" + key.replace("_", "-")
            args += [flag, str(value)]
        else:
            plain_overrides[key] = value
    for key, value in plain_overrides.items():
        args += ["--set", f"{key}={value!r}"]
    return args


def run_one(algo, overrides, dry_run=False):
    cfg = build_config(algo, overrides)
    log_path = cfg["log_path"]
    if log_is_complete(log_path):
        print(f"[skip] {log_path} déjà complet.")
        return
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--algo", algo] + overrides_to_cli(overrides)
    print(f"[run]  {log_path}")
    print(f"       {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def phase2_lambda_mix_ablation(dataset, n_labels, K, seed):
    """PROJECT_SPEC.md §7 Phase 2 : ablation lambda_mix sur EfficientMatch, budget réduit (~2**14),
    1 seule graine, pour figer la valeur avant les runs complets. Chaque valeur reçoit un `tag`
    distinct pour ne pas écraser les logs des autres valeurs testées (même algo/K/seed sinon)."""
    runs = []
    for lambda_mix in (0.5, 1.0, 2.0):
        runs.append(("efficientmatch", {
            "dataset": dataset, "n_labels": n_labels, "seed": seed, "K": K,
            "lambda_mix": lambda_mix, "tag": f"lammix{lambda_mix}",
        }))
    return runs


def phase3_main_comparison(dataset, n_labels, K, seeds, lambda_mix_frozen):
    """PROJECT_SPEC.md §7 Phase 3 : cœur du papier (RQ1+RQ2) -- FixMatch, FlexMatch, MixMatch,
    EfficientMatch (lambda_mix figé par la Phase 2), CIFAR-10, budget de labels principal, K=2**17."""
    runs = []
    for seed in seeds:
        for algo in ("fixmatch", "flexmatch", "mixmatch"):
            runs.append((algo, {"dataset": dataset, "n_labels": n_labels, "seed": seed, "K": K}))
        runs.append(("efficientmatch", {
            "dataset": dataset, "n_labels": n_labels, "seed": seed, "K": K,
            "lambda_mix": lambda_mix_frozen,
        }))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--n-labels", type=int, default=40)
    parser.add_argument("--phase2-K", type=int, default=2 ** 14)
    parser.add_argument("--phase2-seed", type=int, default=0)
    parser.add_argument("--phase3-K", type=int, default=2 ** 17)
    parser.add_argument("--phase3-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--lambda-mix-frozen", type=float, default=None,
                         help="valeur de lambda_mix retenue après lecture des résultats de la Phase 2 "
                              "(python analyze.py --K <phase2-K> ...) -- requis pour lancer la Phase 3")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="affiche la file sans rien lancer")
    args = parser.parse_args()

    queue = []
    if not args.skip_phase2:
        queue += phase2_lambda_mix_ablation(args.dataset, args.n_labels, args.phase2_K, args.phase2_seed)
    if not args.skip_phase3:
        if args.lambda_mix_frozen is None:
            print(
                "ERREUR : --lambda-mix-frozen est requis pour lancer la Phase 3 (c'est le résultat de "
                "l'ablation Phase 2 -- inspectez-le avec `python analyze.py --K <phase2-K> ...` une "
                "fois la Phase 2 terminée). Utilisez --skip-phase3 pour ne lancer que la Phase 2.",
                file=sys.stderr,
            )
            sys.exit(1)
        queue += phase3_main_comparison(
            args.dataset, args.n_labels, args.phase3_K, args.phase3_seeds, args.lambda_mix_frozen
        )

    print(f"File de {len(queue)} run(s) (dry_run={args.dry_run}).\n")
    for algo, overrides in queue:
        run_one(algo, overrides, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
