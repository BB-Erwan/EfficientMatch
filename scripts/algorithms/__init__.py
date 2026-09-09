"""Registre des algorithmes SSL disponibles : nom CLI -> module implémentant

    make_train_step(cfg, device) -> callable

Le callable retourné par `make_train_step` a la signature :
    train_step(model, optimizer, labeled_iter, unlabeled_iter) -> dict[str, float]
"""
from . import fixmatch

ALGORITHMS = {
    "fixmatch": fixmatch,
}
