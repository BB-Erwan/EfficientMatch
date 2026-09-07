"""Fast FixMatch (Chen, Dun & Kyrillidis, 2023/2024) -- FixMatch + Curriculum Batch Size (CBS)."""
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


def compute_curriculum_batch_size(k, cfg):
    """Curriculum Batch Size (CBS), formule B-EXP exacte de Chen, Dun & Kyrillidis (2023/2024) :
    u_t = u_max * (1 - (1 - t/T) / ((1-alpha) + alpha*(1 - t/T))), alpha = 0.7 (sweet spot,
    Table 4 du papier). Croissance lente puis accélérée, atteint u_max exactement à t=T.
    """
    u_max = cfg["mu"] * cfg["B"]
    t_ratio = k / cfg["K"]
    alpha = cfg["cbs_alpha"]
    u_t = u_max * (1 - (1 - t_ratio) / ((1 - alpha) + alpha * (1 - t_ratio)))
    u_t = max(int(u_t), cfg["cbs_min_batch"])
    return min(u_t, u_max)


def estimate_flops_per_iter(model, cfg, device):
    """Mesure réelle des FLOPs, décomposée en un coût FIXE (batch labellisé, taille B constante)
    et un coût MARGINAL par exemple non labellisé (vue faible sans grad + vue forte avec grad).

    Contrairement aux autres algorithmes, le batch non labellisé de Fast FixMatch (Curriculum Batch
    Size) a une taille u_t qui croît de `cbs_min_batch` à `mu*B` au cours de l'entraînement (cf.
    compute_curriculum_batch_size) : les FLOPs par itération ne sont donc PAS constants. Utiliser un
    flops_per_iter unique multiplié par k (comme pour les autres algos) surestimerait
    systématiquement les FLOPs cumulés en début de run et fausserait la métrique
    "FLOPs pour atteindre un seuil de performance", qui est justement l'argument central de CBS
    (cf. PROJECT_SPEC.md §9.6 -- partie la moins vérifiée du code avant ce correctif).

    Les FLOPs scalent linéairement avec la taille de batch pour un réseau convolutif (chaque exemple
    supplémentaire coûte le même nombre de FLOPs) : on mesure donc le coût marginal par exemple à la
    taille de batch max (mu*B) et on le multiplie par u_t réel à chaque itération (cf. flops_for_step).
    """
    model.train()
    B, muB = cfg["B"], cfg["mu"] * cfg["B"]

    # --- Coût fixe : batch labellisé seul (forward + backward, taille B constante) ---
    dummy_x = torch.randn(B, 3, 32, 32, device=device)
    dummy_labels_x = torch.randint(0, cfg["num_classes"], (B,), device=device)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as flop_counter:
        logits_x = model(dummy_x)
        F.cross_entropy(logits_x, dummy_labels_x).backward()
    flops_fixed = flop_counter.get_total_flops()
    model.zero_grad(set_to_none=True)

    # --- Coût marginal : batch non labellisé à taille max (vue faible sans grad + vue forte avec grad) ---
    dummy_u_w = torch.randn(muB, 3, 32, 32, device=device)
    dummy_u_s = torch.randn(muB, 3, 32, 32, device=device)
    dummy_labels_u = torch.randint(0, cfg["num_classes"], (muB,), device=device)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as flop_counter:
        with torch.no_grad():
            _ = model(dummy_u_w)
        logits_u_s = model(dummy_u_s)
        F.cross_entropy(logits_u_s, dummy_labels_u).backward()
    flops_unlabeled_at_max = flop_counter.get_total_flops()
    model.zero_grad(set_to_none=True)

    return {"flops_fixed": flops_fixed, "flops_per_unlabeled": flops_unlabeled_at_max / muB}


def flops_for_step(flops_measurement, step_metrics):
    """FLOPs réels de cette itération = coût fixe + coût marginal * taille réelle du batch (u_t)."""
    return flops_measurement["flops_fixed"] + flops_measurement["flops_per_unlabeled"] * step_metrics["u_t"]


def make_train_step(cfg, augmenter, weak_transform, strong_transform, device):
    """Comparer à `fixmatch.make_train_step` : Étapes 2 (troncature à u_t) et 7 (lambda_u recalé)
    sont les seuls ajouts -- tout le reste (perte supervisée, seuillage, backward, EMA) est identique.
    """
    def train_step(model, ema, optimizer, scaler, k, labeled_iter, unlabeled_iter):
        # --- Étape 1 : batch labellisé (taille fixe l = B) ---
        imgs_x_raw, labels_x = next(labeled_iter)
        imgs_x = augmenter(weak_transform, imgs_x_raw)
        labels_x = labels_x.to(device, non_blocking=True)

        # --- Étape 2 : CURRICULUM BATCH SIZE -- taille du batch non labellisé au temps k ---
        u_t = compute_curriculum_batch_size(k, cfg)
        imgs_u_raw_full, _ = next(unlabeled_iter)   # tiré a taille max (mu*B)
        imgs_u_raw = imgs_u_raw_full[:u_t]          # tronque a u_t elements pour cette iteration

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
            loss_s = F.cross_entropy(logits_x, labels_x)

            # --- Étape 4 : pseudo-étiquetage sur la vue faible (sans gradient) ---
            with torch.no_grad():
                logits_u_w = model(imgs_u_w)
                probs_u_w = F.softmax(logits_u_w, dim=-1)
                max_probs, pseudo_labels = probs_u_w.max(dim=-1)

            # --- Étape 5 : seuil fixe (comme FixMatch -- CPL non inclus dans cette version) ---
            mask = max_probs.ge(cfg["tau"]).float()

            # --- Étape 6 : perte de cohérence faible/forte, filtrée par le masque ---
            logits_u_s = model(imgs_u_s)
            loss_u_per_sample = F.cross_entropy(logits_u_s, pseudo_labels, reduction="none")
            loss_u = (loss_u_per_sample * mask).mean()

            # --- Étape 7 : perte totale, lambda_u recalé sur le ratio de taille de batch courant ---
            lambda_u_t = u_t / cfg["B"]
            loss = loss_s + lambda_u_t * loss_u

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
            "mask_rate": mask.mean().item(), "u_t": u_t, "lambda_u_t": lambda_u_t,
        }

    return train_step
