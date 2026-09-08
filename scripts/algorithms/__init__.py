"""Registre des algorithmes SSL disponibles : nom CLI -> module implémentant

    estimate_flops_per_iter(model, cfg, device) -> float
    make_train_step(cfg, augmenter, weak_transform, strong_transform, device) -> callable

Le callable retourné par `make_train_step` a la signature :
    train_step(model, optimizer, scaler, k, labeled_iter, unlabeled_iter) -> dict[str, float]
"""
from . import efficientmatch, fast_fixmatch, fixmatch, flexmatch, mixmatch

ALGORITHMS = {
    "fixmatch": fixmatch,
    "fast_fixmatch": fast_fixmatch,
    "flexmatch": flexmatch,
    "mixmatch": mixmatch,
    "efficientmatch": efficientmatch,
}
