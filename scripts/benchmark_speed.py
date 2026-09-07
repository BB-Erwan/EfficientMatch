"""Phase 1 (PROJECT_SPEC.md §7) : mesure le temps réel par itération sur la machine cible, pour
calibrer le temps total attendu avant de lancer un run complet. Ne lance AUCUN run réel -- seulement
quelques dizaines d'itérations de chauffe + mesure par algorithme, sur un sous-ensemble réduit des
données (aucune écriture dans ./logs/).

À exécuter sur la machine qui fera réellement tourner les runs (ex. l'A4000) : le temps par itération
dépend du GPU, du pilote CUDA, de la charge concurrente, etc. -- une mesure faite ailleurs ne serait
pas représentative.

Usage :
    python scripts/benchmark_speed.py
    python scripts/benchmark_speed.py --algos fixmatch efficientmatch --bench-iters 50
    python scripts/benchmark_speed.py --K 131072 --n-labels 40
"""
import argparse
import time

import torch

from algorithms import ALGORITHMS
from config import build_config, set_seed
from data import BatchAugmenter, SSLCollate, build_transforms, infinite_loader, load_datasets
from ema import EMA
from models import build_model


def measure_it_per_sec(algo, cfg, warmup_iters, bench_iters):
    """Chronomètre `bench_iters` itérations réelles de train_step (après `warmup_iters` de chauffe,
    non chronométrées -- nécessaires pour que cudnn.benchmark/l'allocateur CUDA se stabilisent)."""
    device = torch.device(cfg["device"])
    set_seed(cfg["seed"])
    if cfg["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True

    weak_transform, strong_transform, eval_transform = build_transforms(cfg)
    augmenter = BatchAugmenter(device, cfg["use_transforms_v2"])
    collate_fn = SSLCollate(cfg["use_transforms_v2"])
    labeled_set, unlabeled_set, _ = load_datasets(cfg, eval_transform)
    labeled_iter = infinite_loader(labeled_set, cfg["B"], cfg, collate_fn, shuffle=True)
    unlabeled_iter = infinite_loader(unlabeled_set, cfg["mu"] * cfg["B"], cfg, collate_fn, shuffle=True)

    model = build_model(cfg, device)
    ema = EMA(model, cfg["ema_decay"])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
        nesterov=cfg["nesterov"], weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler(enabled=cfg["use_amp"])
    train_step = ALGORITHMS[algo].make_train_step(cfg, augmenter, weak_transform, strong_transform, device)

    for k in range(1, warmup_iters + 1):
        train_step(model, ema, optimizer, scaler, k, labeled_iter, unlabeled_iter)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for k in range(warmup_iters + 1, warmup_iters + bench_iters + 1):
        train_step(model, ema, optimizer, scaler, k, labeled_iter, unlabeled_iter)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return bench_iters / elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algos", nargs="+", default=sorted(ALGORITHMS))
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--n-labels", type=int, default=40)
    parser.add_argument("--K", type=int, default=2 ** 17, help="budget utilisé pour l'extrapolation du temps total")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--debug-subset-size", type=int, default=4096,
                         help="sous-ensemble non labellisé utilisé pour la mesure (pas pour les runs réels)")
    args = parser.parse_args()

    label = f"Temps total estimé (K={args.K})"
    header = f"{'Algo':<16}{'it/s':>8}{label:>32}"
    print(header)
    print("-" * len(header))
    for algo in args.algos:
        cfg = build_config(algo, {
            "dataset": args.dataset, "n_labels": args.n_labels,
            "debug_subset_size": args.debug_subset_size,
            "K": args.warmup_iters + args.bench_iters,
        })
        speed = measure_it_per_sec(algo, cfg, args.warmup_iters, args.bench_iters)
        total_hours = args.K / speed / 3600
        print(f"{algo:<16}{speed:>8.2f}{total_hours:>29.2f} h")

    print(
        "\nNote Fast FixMatch (Curriculum Batch Size) : mesure faite à budget artificiellement petit "
        "(warmup+bench_iters), donc proche du pire cas (batch non labellisé quasi maximal) -- le temps "
        "réel sur K complet sera un peu inférieur grâce au CBS."
    )


if __name__ == "__main__":
    main()
