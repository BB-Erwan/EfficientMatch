"""Schedule de learning rate cosine recalé, standard FixMatch/FlexMatch/MixMatch :
lr(k) = lr0 * cos(7*pi*k / (16*K)).
"""
import math


def cosine_schedule(optimizer, k, K):
    base_lr = optimizer.defaults["lr"]
    new_lr = base_lr * math.cos(7 * math.pi * k / (16 * K))
    for group in optimizer.param_groups:
        group["lr"] = max(new_lr, 0.0)
