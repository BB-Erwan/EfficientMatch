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
    python scripts/benchmark_speed.py --n-labels 250 --compile
"""
import argparse
import time

import torch

from algorithms import ALGORITHMS
from algorithms.fast_fixmatch import compute_curriculum_batch_size
from config import build_config, set_seed
from data import BatchAugmenter, SSLCollate, build_transforms, infinite_loader, load_datasets
from ema import EMA
from models import build_model


def _setup(algo, cfg, device):
    """Partie commune : données, modèle, optimiseur -- factorisée pour être réutilisée par la mesure
    standard (débit constant) et par la mesure à taille de batch fixée (Fast FixMatch)."""
    set_seed(cfg["seed"])
    if cfg["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(cfg["matmul_precision"])

    print(f"    données (data_root={cfg['data_root']})...", flush=True)
    weak_transform, strong_transform, eval_transform = build_transforms(cfg)
    augmenter = BatchAugmenter(device, cfg["use_transforms_v2"])
    collate_fn = SSLCollate(cfg["use_transforms_v2"])
    labeled_set, unlabeled_set, _ = load_datasets(cfg, eval_transform)
    labeled_iter = infinite_loader(labeled_set, cfg["B"], cfg, collate_fn, shuffle=True)
    unlabeled_iter = infinite_loader(unlabeled_set, cfg["mu"] * cfg["B"], cfg, collate_fn, shuffle=True)

    print("    modèle...", flush=True)
    model = build_model(cfg, device)
    ema = EMA(model, cfg["ema_decay"]) if cfg["use_ema"] else None
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
        nesterov=cfg["nesterov"], weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler(enabled=cfg["use_amp"] and cfg["amp_dtype"] == "float16")
    train_step = ALGORITHMS[algo].make_train_step(cfg, augmenter, weak_transform, strong_transform, device)
    return train_step, model, ema, optimizer, scaler, labeled_iter, unlabeled_iter


def _run_iters(train_step, model, ema, optimizer, scaler, labeled_iter, unlabeled_iter, k_values, device):
    # ema.update(model) après chaque itération, comme engine.py : c'est un coût réel par itération
    # (une passe sur 100+ tenseurs du state_dict), à inclure dans la mesure de vitesse -- sauté si
    # use_ema=False (cf. _setup, ema vaut alors None).
    for k in k_values:
        train_step(model, optimizer, scaler, k, labeled_iter, unlabeled_iter)
        if ema is not None:
            ema.update(model)
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_it_per_sec(algo, cfg, warmup_iters, bench_iters):
    """Chronomètre `bench_iters` itérations réelles de train_step (après `warmup_iters` de chauffe,
    non chronométrées -- nécessaires pour que cudnn.benchmark/l'allocateur CUDA se stabilisent).
    Utilise une taille de batch CONSTANTE d'une itération à l'autre pour tous les algos sauf Fast
    FixMatch (cf. estimate_fast_fixmatch_hours pour ce cas)."""
    device = torch.device(cfg["device"])
    train_step, model, ema, optimizer, scaler, labeled_iter, unlabeled_iter = _setup(algo, cfg, device)

    print(f"    chauffe ({warmup_iters} it.)...", flush=True)
    _run_iters(train_step, model, ema, optimizer, scaler, labeled_iter, unlabeled_iter,
               range(1, warmup_iters + 1), device)

    print(f"    mesure ({bench_iters} it.)...", flush=True)
    start = time.time()
    _run_iters(train_step, model, ema, optimizer, scaler, labeled_iter, unlabeled_iter,
               range(warmup_iters + 1, warmup_iters + bench_iters + 1), device)
    elapsed = time.time() - start

    return bench_iters / elapsed


def _measure_seconds_per_iter_at_fixed_k(cfg, warmup_iters, bench_iters, fixed_k):
    """Comme measure_it_per_sec, mais fige `k` (donc u_t) à une valeur constante pendant toute la
    mesure : nécessaire pour Fast FixMatch, dont la taille de batch dépend de k -- une taille qui
    change à chaque itération force cuDNN à relancer sa recherche d'algorithme optimal à chaque fois
    (cudnn_benchmark=True), ce qui fausse complètement la mesure (observé : ~25x plus "lent" que les
    autres algos alors que la taille de batch moyenne y était pourtant PLUS PETITE)."""
    device = torch.device(cfg["device"])
    train_step, model, ema, optimizer, scaler, labeled_iter, unlabeled_iter = _setup("fast_fixmatch", cfg, device)

    _run_iters(train_step, model, ema, optimizer, scaler, labeled_iter, unlabeled_iter,
               [fixed_k] * warmup_iters, device)
    start = time.time()
    _run_iters(train_step, model, ema, optimizer, scaler, labeled_iter, unlabeled_iter,
               [fixed_k] * bench_iters, device)
    elapsed = time.time() - start
    return elapsed / bench_iters


def estimate_fast_fixmatch_hours(cfg, warmup_iters, bench_iters):
    """Fast FixMatch a un coût par itération qui varie tout au long du run (Curriculum Batch Size),
    contrairement aux 4 autres algos. On mesure le temps réel à u_t=cbs_min_batch et à u_t=mu*B
    (taille FIXE pendant chaque mesure, cf. _measure_seconds_per_iter_at_fixed_k), on en déduit un
    coût fixe + un coût marginal par exemple non labellisé (le calcul est linéaire en taille de batch,
    même logique que le correctif FLOPs de algorithms/fast_fixmatch.py), puis on intègre sur la vraie
    courbe u_t(k) pour k=1..K -- pas juste une mesure ponctuelle multipliée par K.
    """
    u_min, u_max = cfg["cbs_min_batch"], cfg["mu"] * cfg["B"]

    print(f"    mesure à u_t={u_min} (début de la courbe CBS)...", flush=True)
    t_min = _measure_seconds_per_iter_at_fixed_k(cfg, warmup_iters, bench_iters, fixed_k=1)
    print(f"    mesure à u_t={u_max} (fin de la courbe CBS)...", flush=True)
    t_max = _measure_seconds_per_iter_at_fixed_k(cfg, warmup_iters, bench_iters, fixed_k=cfg["K"])

    seconds_per_unlabeled_sample = (t_max - t_min) / (u_max - u_min)
    seconds_fixed = t_min - seconds_per_unlabeled_sample * u_min

    k_values = torch.arange(1, cfg["K"] + 1, dtype=torch.float64)
    u_t_curve = torch.tensor(
        [compute_curriculum_batch_size(int(k.item()), cfg) for k in k_values], dtype=torch.float64,
    )
    total_seconds = cfg["K"] * seconds_fixed + seconds_per_unlabeled_sample * u_t_curve.sum().item()
    effective_speed = cfg["K"] / total_seconds
    return effective_speed, total_seconds / 3600


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algos", nargs="+", default=sorted(ALGORITHMS))
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--n-labels", type=int, default=250)
    parser.add_argument("--K", type=int, default=2 ** 17, help="budget utilisé pour l'extrapolation du temps total")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--debug-subset-size", type=int, default=4096,
                         help="sous-ensemble non labellisé utilisé pour la mesure (pas pour les runs réels)")
    parser.add_argument("--compile", action="store_true",
                         help="active torch.compile pour la mesure (défaut du config par algo, cf. "
                              "config.py -- reste désactivé pour fast_fixmatch quel que soit ce flag, "
                              "à cause du Curriculum Batch Size). Ajoute un coût de compilation "
                              "ponctuel (~40-70s/algo, absorbé par la chauffe, pas chronométré) ; "
                              "augmentez --warmup-iters si ce n'est pas le cas sur votre machine.")
    args = parser.parse_args()

    label = f"Temps total estimé (K={args.K})"
    header = f"{'Algo':<16}{'it/s':>8}{label:>32}"
    print(header)
    print("-" * len(header))
    for algo in args.algos:
        print(f"[{algo}]", flush=True)
        overrides = {
            "dataset": args.dataset, "n_labels": args.n_labels,
            "debug_subset_size": args.debug_subset_size,
            # Le vrai budget cible, pas warmup+bench_iters : inoffensif pour les 4 algos à taille de
            # batch fixe (K n'intervient pas dans leur train_step), mais indispensable pour Fast
            # FixMatch, dont la courbe de Curriculum Batch Size dépend de k/K (cf.
            # estimate_fast_fixmatch_hours) -- un K artificiellement petit y faussait la mesure.
            "K": args.K,
            # num_workers=0 : évite le coût/risque de démarrage des workers multiprocessing
            # (notoirement lent sur Windows) pour une mesure de quelques secondes.
            "num_workers": 0,
            "persistent_workers": False,
        }
        if not args.compile:
            # Par défaut, ce script ne teste PAS torch.compile : mesure un débit rapide sur
            # quelques dizaines d'itérations, pas la config des vrais runs. Avec --compile, on ne
            # force RIEN ici et on laisse le défaut par algo de config.py s'appliquer (True pour 4
            # algos, False pour fast_fixmatch à cause du Curriculum Batch Size) -- forcer True ici
            # écraserait ce garde-fou spécifique à fast_fixmatch.
            overrides["compile_model"] = False
        cfg = build_config(algo, overrides)
        if algo == "fast_fixmatch":
            speed, total_hours = estimate_fast_fixmatch_hours(cfg, args.warmup_iters, args.bench_iters)
        else:
            speed = measure_it_per_sec(algo, cfg, args.warmup_iters, args.bench_iters)
            total_hours = args.K / speed / 3600
        print(f"{algo:<16}{speed:>8.2f}{total_hours:>29.2f} h")

    print(
        "\nNote Fast FixMatch : 'it/s' est un débit EFFECTIF (K / temps total estimé), pas une valeur "
        "constante -- le coût réel par itération croît tout au long du run à mesure que le Curriculum "
        "Batch Size augmente la taille du batch non labellisé de cbs_min_batch à mu*B."
    )


if __name__ == "__main__":
    main()
