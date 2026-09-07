"""Configuration par défaut, commune à tous les algorithmes SSL + spécifique à chacun.

`BASE_CONFIG` reprend exactement les hyperparamètres des notebooks (mêmes valeurs par défaut).
`ALGO_EXTRA_CONFIG` ajoute les clés propres à chaque algorithme (Mixup, Curriculum Batch Size, etc.).
`DATASET_DEFAULTS` ajoute les clés qui dépendent du dataset choisi (num_classes, weight_decay --
cf. papier : weight_decay=5e-4 pour CIFAR-10/PathMNIST, 1e-3 pour CIFAR-100).
"""
import random

import numpy as np
import torch

BASE_CONFIG = {
    # --- Dataset ---
    "dataset": "cifar10",       # "cifar10" | "cifar100" | "pathmnist"
    "data_root": "./data",
    # n_labels=40 : valeur figée pour CIFAR-10 (régime de faible labellisation, cf. PROJECT_SPEC.md
    # §6/§9.5). À reconfirmer explicitement avant la Phase 3 si un autre budget de labels est retenu.
    "n_labels": 40,
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
    "ema_decay": 0.999,

    # --- Modèle ---
    "depth": 28,
    "widen_factor": 2,

    # --- Budget d'entraînement (protocole réduit, cf. papiers) ---
    "K": 2 ** 17,
    "iters_per_epoch": 1024,
    "eval_every": 512,
    "seed": 0,
    # Étiquette libre incluse dans le nom du fichier de log (ex. "lammix0.5") : indispensable pour
    # distinguer plusieurs runs qui partagent (algo, dataset, n_labels, K, seed) mais diffèrent par
    # un hyperparamètre passé via --set (ex. l'ablation lambda_mix, cf. run_priority_experiments.py)
    # -- sinon ils s'écraseraient tous dans le même fichier.
    "tag": "",

    # --- Early stopping (plateau de l'accuracy EMA, cf. early_stopping.detect_plateau) ---
    # Placeholders (fenêtre=5, seuil=1e-4) : à calibrer empiriquement sur des runs pilotes
    # (Annexe A du papier) avant de figer la valeur pour les runs complets -- cf. PROJECT_SPEC.md §6.
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
        # lambda_mix : NON ENCORE TRANCHÉ (cf. PROJECT_SPEC.md §6/§7 Phase 2). 1.0 est une valeur de
        # travail temporaire -- à remplacer par le résultat de l'ablation {0.5, 1, 2} avant la Phase 3
        # (cf. scripts/run_priority_experiments.py --lambda-mix-frozen).
        "lambda_mix": 1.0,
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
    pour ne jamais écraser le log d'un autre run (notamment entre graines ou entre variantes d'une
    ablation) et pour permettre à run_priority_experiments.py de détecter les runs déjà complétés.
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
