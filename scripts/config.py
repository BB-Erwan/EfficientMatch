"""Configuration par défaut, commune à tous les algorithmes SSL + spécifique à chacun.

`BASE_CONFIG` reprend exactement les hyperparamètres des notebooks (mêmes valeurs par défaut).
`ALGO_EXTRA_CONFIG` ajoute les clés propres à chaque algorithme (Mixup, Curriculum Batch Size, etc.).
"""
import random

import numpy as np
import torch

BASE_CONFIG = {
    # --- Dataset ---
    "dataset": "cifar10",       # "cifar10" | "cifar100"
    "data_root": "./data",
    "n_labels": 40,
    "num_classes": 10,

    # --- Hyperparamètres standards SSL ---
    "B": 64,
    "mu": 7,
    "lr": 0.03,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 5e-4,
    "tau": 0.95,
    "lambda_u": 1.0,
    "ema_decay": 0.999,

    # --- Modèle ---
    "depth": 28,
    "widen_factor": 2,

    # --- Budget d'entraînement (protocole réduit, cf. papiers) ---
    "K": 2 ** 17,
    "iters_per_epoch": 1024,
    "eval_every": 512,
    "seed": 0,

    # --- Early stopping (plateau de l'accuracy EMA, cf. early_stopping.detect_plateau) ---
    "early_stopping": True,
    "es_window": 5,
    "es_slope_threshold": 1e-4,

    # --- Optimisations de vitesse (débrayables) ---
    "use_amp": True,
    "cudnn_benchmark": True,
    "channels_last": True,
    "use_transforms_v2": True,   # True = torchvision.transforms.v2 (batch vectorisé) / False = v1 classique
    "num_workers": 4,
    "persistent_workers": True,
    "pin_memory": True,
    "debug_subset_size": None,
    "compile_model": False,

    # --- Divers ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

ALGO_EXTRA_CONFIG = {
    "fixmatch": {},
    "flexmatch": {},
    "efficientmatch": {
        "alpha_mix": 0.75,      # paramètre de la loi Beta pour le Mixup (repris de MixMatch)
        "lambda_mix": 1.0,      # poids du canal de Mixup filtré
    },
    "fast_fixmatch": {
        "cbs_alpha": 0.7,       # sweet spot rapporté par les auteurs (Table 4 du papier)
        "cbs_min_batch": 8,     # borne basse pour éviter un batch quasi-vide en début d'entraînement
    },
    "mixmatch": {
        "K_aug": 2,             # nombre d'augmentations faibles moyennées pour le pseudo-étiquetage
        "sharpen_T": 0.5,       # température de sharpening
        "alpha_mix": 0.75,      # paramètre de la loi Beta pour le Mixup
        "lambda_u_max": 75.0,   # poids max de la perte non supervisée (papier original CIFAR-10: 75)
        "rampup_length": 16384,  # nombre d'itérations pour le rampup linéaire de lambda_u
    },
}


def build_config(algo_name, overrides=None):
    """Fusionne BASE_CONFIG + les hyperparamètres spécifiques à `algo_name`, puis applique `overrides`."""
    if algo_name not in ALGO_EXTRA_CONFIG:
        raise ValueError(f"Algorithme inconnu : {algo_name} (choix : {sorted(ALGO_EXTRA_CONFIG)})")
    cfg = dict(BASE_CONFIG)
    cfg.update(ALGO_EXTRA_CONFIG[algo_name])
    cfg["algo"] = algo_name
    cfg["log_path"] = f"./logs_{algo_name}.json"
    if overrides:
        cfg.update(overrides)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
