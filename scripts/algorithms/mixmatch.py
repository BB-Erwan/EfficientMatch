"""MixMatch (Berthelot et al., 2019) -- guessing par moyenne de K vues faibles + sharpening + MixUp."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


def sharpen(p, T):
    """Aiguise une distribution de probabilité : p_i^(1/T) / sum_j p_j^(1/T)."""
    p_sharp = p ** (1.0 / T)
    return p_sharp / p_sharp.sum(dim=-1, keepdim=True)


def mixup(x1, p1, x2, p2, alpha):
    """Mélange convexe entre deux paires (image, distribution de probabilité).
    lam est forcé >= 0.5 (convention MixMatch/MixUp) pour que le premier élément de la paire
    domine toujours le mélange -- important pour préserver l'ordre X'/U' après mixage.
    """
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    x = lam * x1 + (1 - lam) * x2
    p = lam * p1 + (1 - lam) * p2
    return x, p


def rampup_lambda_u(k, cfg):
    """Rampup linéaire de lambda_u de 0 à lambda_u_max sur rampup_length itérations.
    Nécessaire dans MixMatch (contrairement à FixMatch/FlexMatch) car en tout début
    d'entraînement les pseudo-étiquettes guessées sont très peu fiables.
    """
    return cfg["lambda_u_max"] * min(1.0, k / cfg["rampup_length"])


def estimate_flops_per_iter(model, cfg, device):
    """Mesure réelle des FLOPs (forward + backward) pour une itération MixMatch."""
    model.train()
    muB_total = cfg["K_aug"] * cfg["mu"] * cfg["B"]
    dummy_x = torch.randn(cfg["B"], 3, 32, 32, device=device)
    dummy_u = torch.randn(muB_total, 3, 32, 32, device=device)
    dummy_labels_x = torch.randint(0, cfg["num_classes"], (cfg["B"],), device=device)
    dummy_labels_u = torch.randint(0, cfg["num_classes"], (muB_total,), device=device)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as flop_counter:
        logits_x = model(dummy_x)
        logits_u = model(dummy_u)
        loss = F.cross_entropy(logits_x, dummy_labels_x) + F.cross_entropy(logits_u, dummy_labels_u)
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def flops_for_step(flops_measurement, step_metrics):
    """FLOPs de cette itération -- constant pour MixMatch (batch de taille fixe)."""
    return flops_measurement


def make_train_step(cfg, augmenter, weak_transform, strong_transform, device):
    """MixMatch n'utilise pas de transform forte -- `strong_transform` est ignoré."""
    def train_step(model, ema, optimizer, scaler, k, labeled_iter, unlabeled_iter):
        # --- Étape 1 : batch labellisé (une seule vue faible) ---
        imgs_x_raw, labels_x_int = next(labeled_iter)
        imgs_x = augmenter(weak_transform, imgs_x_raw)
        labels_x_int = labels_x_int.to(device, non_blocking=True)
        labels_x = F.one_hot(labels_x_int, cfg["num_classes"]).float()

        # --- Étape 2 : batch non labellisé, K augmentations faibles indépendantes ---
        imgs_u_raw, _ = next(unlabeled_iter)
        imgs_u_augs = [augmenter(weak_transform, imgs_u_raw) for _ in range(cfg["K_aug"])]

        if cfg["channels_last"]:
            imgs_x = imgs_x.to(memory_format=torch.channels_last)
            imgs_u_augs = [u.to(memory_format=torch.channels_last) for u in imgs_u_augs]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=cfg["device"], enabled=cfg["use_amp"]):
            # --- Étape 3 : guessing -- moyenne des K prédictions (sans gradient) ---
            with torch.no_grad():
                probs_sum = torch.zeros(imgs_u_augs[0].size(0), cfg["num_classes"], device=device)
                for u_aug in imgs_u_augs:
                    probs_sum += F.softmax(model(u_aug), dim=-1)
                probs_avg = probs_sum / cfg["K_aug"]

                # --- Étape 4 : sharpening ---
                guessed_labels = sharpen(probs_avg, cfg["sharpen_T"])
                # répété K fois pour être aligné avec les K vues augmentées dans la concaténation
                guessed_labels_rep = guessed_labels.repeat(cfg["K_aug"], 1)

            imgs_u_cat = torch.cat(imgs_u_augs, dim=0)  # (K_aug * mu*B, C, H, W)

            # --- Étape 5 : concaténation labellisé + non labellisé, puis mélange ---
            all_imgs = torch.cat([imgs_x, imgs_u_cat], dim=0)
            all_labels = torch.cat([labels_x, guessed_labels_rep], dim=0)
            perm = torch.randperm(all_imgs.size(0), device=device)
            all_imgs_shuffled = all_imgs[perm]
            all_labels_shuffled = all_labels[perm]

            # --- Étape 6 : MixUp entre l'ensemble original et sa version mélangée ---
            mixed_imgs, mixed_labels = mixup(
                all_imgs, all_labels, all_imgs_shuffled, all_labels_shuffled, cfg["alpha_mix"]
            )

            n_x = imgs_x.size(0)  # les n_x premiers éléments correspondent à la partie "labellisée" du mélange
            mixed_x, mixed_labels_x = mixed_imgs[:n_x], mixed_labels[:n_x]
            mixed_u, mixed_labels_u = mixed_imgs[n_x:], mixed_labels[n_x:]

            logits_mixed_x = model(mixed_x)
            logits_mixed_u = model(mixed_u)

            # --- Étape 7 : pertes ---
            loss_x = -(mixed_labels_x * F.log_softmax(logits_mixed_x, dim=-1)).sum(dim=-1).mean()
            loss_u = F.mse_loss(F.softmax(logits_mixed_u, dim=-1), mixed_labels_u)
            lambda_u = rampup_lambda_u(k, cfg)
            loss = loss_x + lambda_u * loss_u

        # --- Étape 8 : backward + optimisation ---
        if cfg["use_amp"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # --- Étape 9 : mise à jour EMA ---
        ema.update(model)

        return {
            "loss": loss.item(), "loss_x": loss_x.item(), "loss_u": loss_u.item(),
            "lambda_u": lambda_u,
        }

    return train_step
