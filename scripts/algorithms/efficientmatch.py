"""EfficientMatch -- FixMatch + canal de Mixup filtré par le masque de confiance dur."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


def mixup_pair(x1, p1, x2, p2, alpha):
    """Mélange convexe entre deux paires (image, distribution de probabilité / label one-hot).
    lam est forcé >= 0.5 (convention MixMatch/MixUp).
    """
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)
    x = lam * x1 + (1 - lam) * x2
    p = lam * p1 + (1 - lam) * p2
    return x, p


def estimate_flops_per_iter(model, cfg, device):
    """Mesure réelle des FLOPs (forward + backward). Inclut le forward supplémentaire du canal de
    Mixup filtré (taille B + mu*B) par rapport à FixMatch.
    """
    model.train()
    B, muB = cfg["B"], cfg["mu"] * cfg["B"]
    dummy_x = torch.randn(B, 3, 32, 32, device=device)
    dummy_u_w = torch.randn(muB, 3, 32, 32, device=device)
    dummy_u_s = torch.randn(muB, 3, 32, 32, device=device)
    dummy_mixed = torch.randn(B + muB, 3, 32, 32, device=device)
    dummy_labels_x = torch.randint(0, cfg["num_classes"], (B,), device=device)
    dummy_labels_u = torch.randint(0, cfg["num_classes"], (muB,), device=device)
    dummy_labels_mix = torch.randint(0, cfg["num_classes"], (B + muB,), device=device)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as flop_counter:
        logits_x = model(dummy_x)
        with torch.no_grad():
            _ = model(dummy_u_w)
        logits_u_s = model(dummy_u_s)
        logits_mixed = model(dummy_mixed)
        loss = (F.cross_entropy(logits_x, dummy_labels_x)
                + F.cross_entropy(logits_u_s, dummy_labels_u)
                + F.cross_entropy(logits_mixed, dummy_labels_mix))
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def make_train_step(cfg, augmenter, weak_transform, strong_transform, device):
    """Comparer à `fixmatch.make_train_step` : Étapes 1-5 et 8-9 identiques ; l'Étape 6 (Mixup
    filtré) est le seul ajout, inséré avant le calcul de la perte totale (Étape 7).
    """
    def train_step(model, ema, optimizer, scaler, k, labeled_iter, unlabeled_iter):
        # --- Étape 1 : batch labellisé ---
        imgs_x_raw, labels_x_int = next(labeled_iter)
        imgs_x = augmenter(weak_transform, imgs_x_raw)
        labels_x_int = labels_x_int.to(device, non_blocking=True)
        labels_x_onehot = F.one_hot(labels_x_int, cfg["num_classes"]).float()

        # --- Étape 2 : batch non labellisé, deux vues (faible / forte) ---
        imgs_u_raw, _ = next(unlabeled_iter)
        imgs_u_w = augmenter(weak_transform, imgs_u_raw)
        imgs_u_s = augmenter(strong_transform, imgs_u_raw)

        if cfg["channels_last"]:
            imgs_x = imgs_x.to(memory_format=torch.channels_last)
            imgs_u_w = imgs_u_w.to(memory_format=torch.channels_last)
            imgs_u_s = imgs_u_s.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=cfg["device"], enabled=cfg["use_amp"]):
            # --- Étape 3 : perte supervisée ---
            logits_x = model(imgs_x)
            loss_s = F.cross_entropy(logits_x, labels_x_int)

            # --- Étape 4 : pseudo-étiquetage sur la vue faible (sans gradient) + masque dur ---
            with torch.no_grad():
                logits_u_w = model(imgs_u_w)
                probs_u_w = F.softmax(logits_u_w, dim=-1)
                max_probs, pseudo_labels = probs_u_w.max(dim=-1)
                mask = max_probs.ge(cfg["tau"]).float()
                pseudo_labels_onehot = F.one_hot(pseudo_labels, cfg["num_classes"]).float()

            # --- Étape 5 : perte FixMatch classique (cohérence faible/forte filtrée) ---
            logits_u_s = model(imgs_u_s)
            loss_u_per_sample = F.cross_entropy(logits_u_s, pseudo_labels, reduction="none")
            loss_u = (loss_u_per_sample * mask).mean()

            # --- Étape 6 : MIXUP FILTRÉ (contribution EfficientMatch) ---
            mask_x = torch.ones(imgs_x.size(0), device=device)  # masque toujours = 1 pour le batch labellisé

            all_imgs = torch.cat([imgs_x, imgs_u_w], dim=0)
            all_labels = torch.cat([labels_x_onehot, pseudo_labels_onehot], dim=0)
            all_masks = torch.cat([mask_x, mask], dim=0)

            perm = torch.randperm(all_imgs.size(0), device=device)
            all_imgs_shuffled = all_imgs[perm]
            all_labels_shuffled = all_labels[perm]
            all_masks_shuffled = all_masks[perm]

            mixed_imgs, mixed_labels = mixup_pair(
                all_imgs, all_labels, all_imgs_shuffled, all_labels_shuffled, cfg["alpha_mix"]
            )
            # masque dur : une paire ne contribue que si SES DEUX composants sont fiables
            pair_mask = all_masks * all_masks_shuffled

            logits_mixed = model(mixed_imgs)
            loss_mix_per_sample = -(mixed_labels * F.log_softmax(logits_mixed, dim=-1)).sum(dim=-1)
            loss_mix = (loss_mix_per_sample * pair_mask).sum() / (pair_mask.sum() + 1e-8)

            # --- Étape 7 : perte totale ---
            loss = loss_s + cfg["lambda_u"] * loss_u + cfg["lambda_mix"] * loss_mix

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
            "loss": loss.item(), "loss_s": loss_s.item(), "loss_u": loss_u.item(),
            "loss_mix": loss_mix.item(), "mask_rate": mask.mean().item(),
            "pair_mask_rate": pair_mask.mean().item(),
        }

    return train_step
