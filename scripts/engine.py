"""Boucle d'entraînement générique, partagée par tous les algorithmes SSL.

Assemble données, modèle, optimiseur, scheduler et logging JSON, puis délègue la logique propre à
chaque algorithme aux fonctions `estimate_flops_per_iter` / `make_train_step` du module `algorithms.<algo>`.
"""
import json
import time

import torch
from torch.utils.data import DataLoader

from config import set_seed
from data import BatchAugmenter, SSLCollate, build_transforms, infinite_loader, load_datasets
from ema import EMA
from evaluate import evaluate
from models import build_model
from schedule import cosine_schedule


def run_experiment(cfg, algo_module):
    set_seed(cfg["seed"])
    if cfg["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    device = torch.device(cfg["device"])

    weak_transform, strong_transform, eval_transform = build_transforms(cfg)
    augmenter = BatchAugmenter(device, cfg["use_transforms_v2"])
    collate_fn = SSLCollate(cfg["use_transforms_v2"])

    labeled_set, unlabeled_set, test_set = load_datasets(cfg, eval_transform)
    labeled_iter = infinite_loader(labeled_set, cfg["B"], cfg, collate_fn, shuffle=True)
    unlabeled_iter = infinite_loader(unlabeled_set, cfg["mu"] * cfg["B"], cfg, collate_fn, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=cfg["num_workers"])

    model = build_model(cfg, device)
    eval_model = build_model(cfg, device)
    ema = EMA(model, cfg["ema_decay"])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
        nesterov=cfg["nesterov"], weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler(enabled=cfg["use_amp"])

    flops_per_iter = algo_module.estimate_flops_per_iter(model, cfg, device)
    print(f"FLOPs (mesurés) par itération : {flops_per_iter:.3e}")
    print(f"FLOPs totaux estimés : {flops_per_iter * cfg['K']:.3e}")
    print(f"Budget total : {cfg['K']} itérations")

    train_step = algo_module.make_train_step(cfg, augmenter, weak_transform, strong_transform, device)

    logs = []
    start_time = time.time()
    model.train()
    for k in range(1, cfg["K"] + 1):
        cosine_schedule(optimizer, k, cfg["K"])
        step_metrics = train_step(model, ema, optimizer, scaler, k, labeled_iter, unlabeled_iter)

        if k % cfg["eval_every"] == 0 or k == cfg["K"]:
            ema.copy_to(eval_model)
            acc = evaluate(eval_model, test_loader, device)
            elapsed = time.time() - start_time
            log_entry = {
                "iteration": k, "elapsed_seconds": elapsed,
                "cumulative_flops": flops_per_iter * k, "eval_accuracy": acc, **step_metrics,
            }
            logs.append(log_entry)
            extra = " ".join(f"{name}={value:.3f}" for name, value in step_metrics.items() if name != "loss")
            print(f"[iter {k:>7}/{cfg['K']}] acc={acc:.4f} loss={step_metrics['loss']:.4f} "
                  f"{extra} elapsed={elapsed / 60:.1f}min")
            with open(cfg["log_path"], "w") as f:
                json.dump({"config": cfg, "logs": logs}, f, indent=2)

    print("Entraînement terminé.")
    return logs
