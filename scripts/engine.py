"""Boucle d'entraînement FixMatch : assemble données, modèle, optimiseur, scheduler et logging JSON."""
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import set_seed
from data import build_transforms, infinite_loader, load_datasets
from early_stopping import detect_plateau
from ema import EMA
from evaluate import evaluate
from models import build_model
from schedule import cosine_schedule


def run_experiment(cfg, algo_module):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    weak_transform, strong_transform, eval_transform = build_transforms(cfg)
    labeled_set, unlabeled_set, test_set = load_datasets(cfg, weak_transform, strong_transform, eval_transform)
    labeled_iter = infinite_loader(labeled_set, cfg["B"], cfg, shuffle=True)
    unlabeled_iter = infinite_loader(unlabeled_set, cfg["mu"] * cfg["B"], cfg, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=cfg["num_workers"])

    model = build_model(cfg, device)
    if cfg["use_ema"]:
        eval_model = build_model(cfg, device)
        ema = EMA(model, cfg["ema_decay"])
    else:
        eval_model = model
        ema = None
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
        nesterov=cfg["nesterov"], weight_decay=cfg["weight_decay"],
    )

    print(f"Budget total : {cfg['K']} itérations")

    train_step = algo_module.make_train_step(cfg, device)

    Path(cfg["log_path"]).parent.mkdir(parents=True, exist_ok=True)

    logs = []
    acc_history = []
    start_time = time.time()
    model.train()
    stopped_early = False
    last_k = 0
    for k in range(1, cfg["K"] + 1):
        cosine_schedule(optimizer, k, cfg["K"])
        step_metrics = train_step(model, optimizer, labeled_iter, unlabeled_iter)
        if ema is not None:
            ema.update(model)
        last_k = k

        if k % cfg["eval_every"] == 0 or k == cfg["K"]:
            if ema is not None:
                ema.copy_to(eval_model)
            acc = evaluate(eval_model, test_loader, device)
            elapsed = time.time() - start_time
            log_entry = {
                "iteration": k, "elapsed_seconds": elapsed,
                "eval_accuracy": acc, **step_metrics,
            }
            logs.append(log_entry)
            extra = " ".join(f"{name}={value:.3f}" for name, value in step_metrics.items() if name != "loss")
            print(f"[iter {k:>7}/{cfg['K']}] acc={acc:.4f} loss={step_metrics['loss']:.4f} "
                  f"{extra} elapsed={elapsed / 60:.1f}min")
            with open(cfg["log_path"], "w") as f:
                json.dump({"config": cfg, "logs": logs}, f, indent=2)

            acc_history.append(acc)
            if cfg["early_stopping"]:
                is_plateau, slope = detect_plateau(acc_history, cfg["es_window"], cfg["es_slope_threshold"])
                if is_plateau:
                    print(f"Plateau détecté (pente={slope:.2e} < seuil={cfg['es_slope_threshold']:.2e}) "
                          f"-- arrêt anticipé à l'itération {k}.")
                    stopped_early = True
                    break

    # Marqueur de complétude (distinct d'un run interrompu/crashé en cours de route) : utilisé par
    # analyze.py pour savoir jusqu'à quelle itération reporter la dernière valeur observée
    # (cf. PROJECT_SPEC.md §4).
    with open(cfg["log_path"], "w") as f:
        json.dump({
            "config": cfg, "logs": logs, "status": "completed",
            "stopped_early": stopped_early, "last_iteration": last_k,
        }, f, indent=2)

    print("Entraînement terminé.")
    return logs
