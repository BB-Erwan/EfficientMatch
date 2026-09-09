"""Configuration par défaut pour l'entraînement FixMatch.

`BASE_CONFIG` reprend les hyperparamètres standards du protocole FixMatch.
`DATASET_DEFAULTS` ajoute les clés qui dépendent du dataset choisi (num_classes, weight_decay --
cf. papier : weight_decay=5e-4 pour CIFAR-10/PathMNIST, 1e-3 pour CIFAR-100).
"""
import os
import random

import numpy as np
import torch

# Racine du projet (parent de scripts/), calculée depuis l'emplacement de ce fichier -- PAS depuis le
# répertoire courant. "./data" dépendrait du dossier depuis lequel la commande est lancée (racine du
# projet vs scripts/ vs ailleurs dans un IDE), et pointerait donc vers un dossier différent -- voire
# vide -- selon le cwd, provoquant un retéléchargement complet du dataset à chaque fois qu'on ne
# lance pas exactement depuis le même endroit.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_CONFIG = {
    # --- Dataset ---
    "dataset": "cifar10",       # "cifar10" | "cifar100" | "pathmnist"
    "data_root": os.path.join(_PROJECT_ROOT, "data"),
    # n_labels=250 : valeur figée pour CIFAR-10 (décidée après le benchmark de Phase 1, cf.
    # BENCHMARK_RESULTS.md -- remplace le placeholder à 40 de PROJECT_SPEC.md §6/§9.5).
    "n_labels": 250,
    "num_classes": 10,          # écrasé automatiquement selon `dataset`, cf. DATASET_DEFAULTS

    # --- Hyperparamètres standards SSL ---
    "B": 64,
    "mu": 7,
    "lr": 0.03,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 5e-4,        # écrasé automatiquement selon `dataset`, cf. DATASET_DEFAULTS
    "tau": 0.95,
    "lambda_u": 1.0,
    # EMA débrayable : "use_ema": False -> engine.py évalue directement les poids en cours
    # d'entraînement (pas d'eval_model séparé, pas de copie de state_dict à chaque évaluation).
    "use_ema": True,
    "ema_decay": 0.999,

    # --- Modèle ---
    "depth": 28,
    "widen_factor": 2,

    # --- Budget d'entraînement (protocole réduit, cf. papiers) ---
    "K": 2 ** 17,
    "iters_per_epoch": 1024,
    "eval_every": 512,
    "seed": 0,
    # Étiquette libre incluse dans le nom du fichier de log (ex. "run1") : indispensable pour
    # distinguer plusieurs runs qui partagent (algo, dataset, n_labels, K, seed) mais diffèrent par
    # un hyperparamètre passé via --set -- sinon ils s'écraseraient tous dans le même fichier.
    "tag": "",

    # --- Early stopping (plateau de l'accuracy EMA, cf. early_stopping.detect_plateau) ---
    "early_stopping": True,
    "es_window": 5,
    "es_slope_threshold": 1e-4,

    # --- Chargement des données ---
    "num_workers": 2,
    "debug_subset_size": None,

    # --- Divers ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

ALGO_EXTRA_CONFIG = {
    "fixmatch": {},
}

# Hyperparamètres qui dépendent du dataset choisi (cf. papier, Table des hyperparamètres).
# Appliqués automatiquement dans build_config() ; restent toutefois surchargeables explicitement
# (--num-classes / --weight-decay en CLI, ou `overrides` en appel programmatique) si besoin.
DATASET_DEFAULTS = {
    "cifar10":   {"num_classes": 10,  "weight_decay": 5e-4},
    "cifar100":  {"num_classes": 100, "weight_decay": 1e-3},
    "pathmnist": {"num_classes": 9,   "weight_decay": 5e-4},
}


def compute_log_path(cfg):
    """Chemin de log déterministe : un fichier distinct par (algo, dataset, n_labels, K, seed, tag),
    pour ne jamais écraser le log d'un autre run (notamment entre graines).
    """
    suffix = f"_{cfg['tag']}" if cfg.get("tag") else ""
    return (
        f"./logs/{cfg['algo']}_{cfg['dataset']}_n{cfg['n_labels']}_K{cfg['K']}_seed{cfg['seed']}{suffix}.json"
    )


def build_config(algo_name, overrides=None):
    """Fusionne BASE_CONFIG + les hyperparamètres spécifiques à `algo_name` + ceux spécifiques au
    dataset choisi, puis applique `overrides` (qui a toujours le dernier mot, y compris sur
    num_classes/weight_decay si l'appelant veut explicitement les forcer).
    """
    if algo_name not in ALGO_EXTRA_CONFIG:
        raise ValueError(f"Algorithme inconnu : {algo_name} (choix : {sorted(ALGO_EXTRA_CONFIG)})")
    overrides = dict(overrides) if overrides else {}

    cfg = dict(BASE_CONFIG)
    cfg.update(ALGO_EXTRA_CONFIG[algo_name])

    dataset = overrides.get("dataset", cfg["dataset"])
    cfg.update(DATASET_DEFAULTS.get(dataset, {}))
    cfg["dataset"] = dataset

    cfg["algo"] = algo_name
    cfg.update(overrides)
    cfg["log_path"] = compute_log_path(cfg)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
