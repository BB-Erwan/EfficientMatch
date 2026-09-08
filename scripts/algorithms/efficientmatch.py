"""EfficientMatch -- FixMatch + canal de Mixup filtré par le masque de confiance dur.

L'algorithme (forward fusionné, canal Mixup, gradient clipping) est une transcription fidèle de la
référence validée dans `efficientmatch_debug.py`, branchée sur l'infrastructure commune du dépôt :
schedule de LR recalé (cf. engine.py + schedule.py), augmentations RandAugment(2,10)+RandomErasing de
`data.py`, AMP/channels_last/torch.compile, hyperparamètres pilotés par `cfg` (tau, lambda_u,
lambda_mix, alpha_mix). Seul l'ALGORITHME vient du script de debug -- pas ses réglages ad hoc
(lr/schedule/augmentations propres au script, qui restent ceux, plus faibles/différents, du debug).
"""
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from config import AMP_DTYPES


def estimate_flops_per_iter(model, cfg, device):
    """Mesure réelle des FLOPs (forward + backward), reflétant le regroupement réel des forwards de
    `make_train_step` : (1) vue faible seule, sans gradient (pseudo-étiquetage) ; (2) forward FUSIONNÉ
    labellisé+cohérence forte (taille B + mu*B) ; (3) forward fusionné du canal Mixup (taille B + mu*B).
    """
    model.train()
    B, muB = cfg["B"], cfg["mu"] * cfg["B"]
    dummy_u_w = torch.randn(muB, 3, 32, 32, device=device)
    dummy_fused = torch.randn(B + muB, 3, 32, 32, device=device)
    dummy_mixed = torch.randn(B + muB, 3, 32, 32, device=device)
    dummy_labels_fused = torch.randint(0, cfg["num_classes"], (B + muB,), device=device)
    dummy_labels_mixed = torch.randint(0, cfg["num_classes"], (B + muB,), device=device)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as flop_counter:
        with torch.no_grad():
            _ = model(dummy_u_w)
        logits_fused = model(dummy_fused)
        logits_mixed = model(dummy_mixed)
        loss = (F.cross_entropy(logits_fused, dummy_labels_fused)
                + F.cross_entropy(logits_mixed, dummy_labels_mixed))
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def flops_for_step(flops_measurement, step_metrics):
    """FLOPs de cette itération -- constant pour EfficientMatch (toutes les tailles de batch,
    y compris celle du canal Mixup, sont fixes)."""
    return flops_measurement


def make_train_step(cfg, augmenter, weak_transform, strong_transform, device):
    """Comparer à `fixmatch.make_train_step` : la perte supervisée et la perte de cohérence sont
    calculées à partir d'un SEUL forward fusionné (au lieu de deux forwards séparés) pour que
    BatchNorm voie les mêmes statistiques de batch que dans `efficientmatch_debug.py` -- un forward
    séparé changerait les statistiques BN, donc la dynamique d'entraînement, pas juste la vitesse.
    Le canal Mixup (aligné sur `efficientmatch_debug.py`, et non sur une formulation MixUp générique
    à lambda unique par batch) est le second ajout :
    - lambda PAR ÉCHANTILLON (un tirage Beta distinct par élément, pas un scalaire unique pour tout
      le batch) -- un tirage pour la partie labellisée, un autre pour la partie non labellisée ;
    - masque de confiance appliqué via le partenaire mélangé UNIQUEMENT (pas le produit avec le
      masque de l'ancre elle-même) ;
    - cible dure (le vrai label) pour la partie labellisée, cible SOFT (distribution softmax
      complète `probs_u_w`, pas le pseudo-label dur) pour la partie non labellisée ;
    - normalisation en DEUX moyennes séparées (partie labellisée / partie non labellisée, sommées),
      pas une moyenne unique sur le nombre de paires confiantes.
    Gradient clipping (`cfg["grad_clip"]`, désactivable) après le backward -- comme la référence,
    contrairement aux autres algos du dépôt qui n'en font pas.
    """
    beta_dist = torch.distributions.Beta(
        torch.tensor(cfg["alpha_mix"], device=device, dtype=torch.float32),
        torch.tensor(cfg["alpha_mix"], device=device, dtype=torch.float32),
    )

    def train_step(model, optimizer, scaler, k, labeled_iter, unlabeled_iter):
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

        with torch.autocast(device_type=cfg["device"], enabled=cfg["use_amp"], dtype=AMP_DTYPES[cfg["amp_dtype"]]):
            # --- Étape 3 : pseudo-étiquetage sur la vue faible (sans gradient) + masque dur ---
            with torch.no_grad():
                logits_u_w = model(imgs_u_w)
                # .float() explicite : sous autocast, la sortie de softmax n'est pas garantie fp32
                # selon la plateforme/dtype (fp32 forcé sur CUDA, pas toujours sur CPU) -- probs_u_w
                # sert de cible SOFT dans torch.lerp plus bas (Étape 5), qui exige un dtype identique
                # à all_labels_shuffled (toujours fp32, cf. labels_x_onehot/pseudo_labels_onehot).
                probs_u_w = F.softmax(logits_u_w, dim=-1).float()
                max_probs, pseudo_labels = probs_u_w.max(dim=-1)
                mask = max_probs.ge(cfg["tau"]).float()
                pseudo_labels_onehot = F.one_hot(pseudo_labels, cfg["num_classes"]).float()

            # --- Étape 4 : forward FUSIONNÉ labellisé + cohérence forte (cf. docstring) ---
            B = imgs_x.size(0)
            all_logits = model(torch.cat([imgs_x, imgs_u_s], dim=0))
            logits_x = all_logits[:B]
            logits_u_s = all_logits[B:]

            loss_s = F.cross_entropy(logits_x, labels_x_int)
            loss_u_per_sample = F.cross_entropy(logits_u_s, pseudo_labels, reduction="none")
            loss_u = (loss_u_per_sample * mask).mean()

            # --- Étape 5 : MIXUP FILTRÉ (aligné sur efficientmatch_debug.py) ---
            mask_x = torch.ones(B, device=device)

            all_imgs = torch.cat([imgs_x, imgs_u_w], dim=0)
            all_labels = torch.cat([labels_x_onehot, pseudo_labels_onehot], dim=0)
            all_masks = torch.cat([mask_x, mask], dim=0)

            perm = torch.randperm(all_imgs.size(0), device=device)
            all_imgs_shuffled = all_imgs[perm]
            all_labels_shuffled = all_labels[perm]
            all_masks_shuffled = all_masks[perm]

            # Lambda par échantillon (pas un scalaire par batch), forcé >= 0.5 pour que l'ancre
            # d'origine (imgs_x / imgs_u_w) domine toujours le mélange.
            lam_x = beta_dist.sample((B,))
            lam_u = beta_dist.sample((imgs_u_w.size(0),))
            lam_x = torch.maximum(lam_x, 1 - lam_x)
            lam_u = torch.maximum(lam_u, 1 - lam_u)

            mixed_x_imgs = torch.lerp(all_imgs_shuffled[:B], imgs_x, lam_x.view(-1, 1, 1, 1))
            mixed_u_imgs = torch.lerp(all_imgs_shuffled[B:], imgs_u_w, lam_u.view(-1, 1, 1, 1))

            # Cible dure pour la partie labellisée, SOFT (probs_u_w) pour la partie non labellisée.
            mixed_x_labels = torch.lerp(all_labels_shuffled[:B], labels_x_onehot, lam_x.view(-1, 1))
            mixed_u_labels = torch.lerp(all_labels_shuffled[B:], probs_u_w, lam_u.view(-1, 1))

            all_mixed_imgs = torch.cat([mixed_x_imgs, mixed_u_imgs], dim=0)
            logits_mixed = model(all_mixed_imgs)
            logits_x_mixed = logits_mixed[:B]
            logits_u_mixed = logits_mixed[B:]

            # Masque du partenaire mélangé uniquement (pas de produit avec le masque de l'ancre).
            mask_x_shuffled = all_masks_shuffled[:B]
            mask_u_shuffled = all_masks_shuffled[B:]

            # Deux moyennes séparées (labellisé / non labellisé), sommées -- pas une moyenne unique
            # sur le nombre de paires confiantes.
            loss_mix = (
                (mask_x_shuffled * F.cross_entropy(logits_x_mixed, mixed_x_labels, reduction="none")).mean()
                + (mask_u_shuffled * F.cross_entropy(logits_u_mixed, mixed_u_labels, reduction="none")).mean()
            )

            # --- Étape 6 : perte totale ---
            loss = loss_s + cfg["lambda_u"] * loss_u + cfg["lambda_mix"] * loss_mix

        # --- Étape 7 : backward + gradient clipping (spécifique à EfficientMatch) + optimisation ---
        if cfg["use_amp"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_max_norm"])
            optimizer.step()

        return {
            "loss": loss.item(), "loss_s": loss_s.item(), "loss_u": loss_u.item(),
            "loss_mix": loss_mix.item(), "mask_rate": mask.mean().item(),
            "mixup_mask_rate": all_masks_shuffled.mean().item(),
        }

    return train_step
