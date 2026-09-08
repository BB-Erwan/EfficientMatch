"""Configuration par défaut, commune à tous les algorithmes SSL + spécifique à chacun.

`BASE_CONFIG` reprend exactement les hyperparamètres des notebooks (mêmes valeurs par défaut).
`ALGO_EXTRA_CONFIG` ajoute les clés propres à chaque algorithme (Mixup, Curriculum Batch Size, etc.).
`DATASET_DEFAULTS` ajoute les clés qui dépendent du dataset choisi (num_classes, weight_decay --
cf. papier : weight_decay=5e-4 pour CIFAR-10/PathMNIST, 1e-3 pour CIFAR-100).
"""
import os
import random

# Contourne un bug Windows : le cache de compilation triton/inductor écrit par défaut dans
# %TEMP%\torchinductor_<user>\... ; combiné aux sous-dossiers de hash de triton, ce chemin dépasse
# souvent la limite de 260 caractères de Windows, provoquant un FileNotFoundError silencieux lors de
# la toute première compilation (vérifié sur cette machine : nom d'utilisateur long -> déjà tronqué
# en 8.3 par Windows dans %TEMP%). Un chemin court à la racine du disque système évite le problème.
# Ne s'applique que sur Windows, et seulement si l'utilisateur n'a pas déjà fixé ces variables lui-même.
if os.name == "nt":
    _cache_root = os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "tc")
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, "inductor"))
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)

import numpy as np
import torch

# dtype réel utilisé par torch.autocast(dtype=...) -- cfg ne stocke que la clé string ("float16" |
# "bfloat16"), jamais l'objet torch.dtype, car cfg est sérialisé en JSON dans les logs.
AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}

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
    # Précision des matmuls float32 (torch.set_float32_matmul_precision) : "high" active TF32 sur
    # Ampere+ (ex. A4000) pour les matmuls hors autocast (poids maîtres, étape d'optimiseur) --
    # gain de vitesse quasi gratuit, perte de précision négligeable pour ce protocole.
    # Choix : "highest" (fp32 complet) | "high" (TF32) | "medium".
    "matmul_precision": "high",
    "use_amp": True,
    # dtype utilisé sous torch.autocast : "float16" (nécessite le GradScaler, gradients pouvant
    # sous-flotter) ou "bfloat16" (même plage d'exposant que fp32, pas de sous-flottement -> le
    # GradScaler est automatiquement désactivé pour ce choix, cf. engine.py/benchmark_speed.py).
    "amp_dtype": "bfloat16",
    "cudnn_benchmark": True,
    "channels_last": True,
    "use_transforms_v2": True,   # True = torchvision.transforms.v2 (batch vectorisé) / False = v1 classique
    "num_workers": 4,
    "persistent_workers": True,
    "pin_memory": True,
    "debug_subset_size": None,
    # True par défaut : formes de batch fixes pour 4 des 5 algos, torch.compile amortit son coût de
    # compilation initial sur les dizaines de milliers d'itérations d'un run complet. Forcé à False
    # spécifiquement pour fast_fixmatch ci-dessous (cf. ALGO_EXTRA_CONFIG) -- son Curriculum Batch
    # Size change la taille du batch non labellisé à chaque itération, ce qui déclencherait une
    # recompilation quasi permanente au lieu d'une accélération.
    "compile_model": True,

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
        # Garde-fou : la taille de batch non labellisé varie à chaque itération (CBS), donc
        # torch.compile recompilerait en permanence au lieu d'accélérer -- cf. BASE_CONFIG.
        # Reste explicitement surchargeable (--compile-model / --set compile_model=True) si vous
        # voulez tester `dynamic=True` vous-même, mais ce n'est plus la valeur par défaut.
        "compile_model": False,
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
