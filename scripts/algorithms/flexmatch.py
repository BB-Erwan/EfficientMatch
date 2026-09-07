"""FlexMatch (Zhang et al., 2021) -- FixMatch + seuillage adaptatif par classe (Curriculum Pseudo Labeling)."""
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


def estimate_flops_per_iter(model, cfg, device):
    """Mesure réelle des FLOPs (forward + backward) pour une itération FlexMatch."""
    model.train()
    B, muB = cfg["B"], cfg["mu"] * cfg["B"]
    dummy_x = torch.randn(B, 3, 32, 32, device=device)
    dummy_u_w = torch.randn(muB, 3, 32, 32, device=device)
    dummy_u_s = torch.randn(muB, 3, 32, 32, device=device)
    dummy_labels_x = torch.randint(0, cfg["num_classes"], (B,), device=device)
    dummy_labels_u = torch.randint(0, cfg["num_classes"], (muB,), device=device)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as flop_counter:
        logits_x = model(dummy_x)
        with torch.no_grad():
            _ = model(dummy_u_w)
        logits_u_s = model(dummy_u_s)
        loss = F.cross_entropy(logits_x, dummy_labels_x) + F.cross_entropy(logits_u_s, dummy_labels_u)
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def flops_for_step(flops_measurement, step_metrics):
    """FLOPs de cette itération -- constant pour FlexMatch (batch de taille fixe)."""
    return flops_measurement


def make_train_step(cfg, augmenter, weak_transform, strong_transform, device):
    """Pour basculer vers FixMatch : remplacer le bloc "SEUIL ADAPTATIF" par un seuil fixe
    `cfg['tau']` et supprimer la mise à jour de `class_counts` -- rien d'autre ne change.
    """
    class_counts = {c: 0 for c in range(cfg["num_classes"])}  # état persistant entre itérations

    def train_step(model, ema, optimizer, scaler, k, labeled_iter, unlabeled_iter):
        # --- Étape 1 : batch labellisé ---
        imgs_x_raw, labels_x = next(labeled_iter)
        imgs_x = augmenter(weak_transform, imgs_x_raw)
        labels_x = labels_x.to(device, non_blocking=True)

        # --- Étape 2 : batch non labellisé, deux vues (faible / forte) ---
        imgs_u_raw, _ = next(unlabeled_iter)  # le vrai label n'est jamais utilisé pendant l'entraînement
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

            # --- Étape 5 : SEUIL ADAPTATIF (spécifique FlexMatch -- Curriculum Pseudo Labeling) ---
            with torch.no_grad():
                confident_mask_raw = max_probs.ge(cfg["tau"])
                for c in range(cfg["num_classes"]):
                    class_counts[c] += ((pseudo_labels == c) & confident_mask_raw).sum().item()

                max_count = max(max(class_counts.values()), 1)  # évite division par zéro au tout début
                beta_t = {c: class_counts[c] / max_count for c in range(cfg["num_classes"])}

                tau_per_class = torch.tensor(
                    [beta_t[c] / (2 - beta_t[c]) * cfg["tau"] for c in range(cfg["num_classes"])],
                    device=device,
                )

                # masque final : la pseudo-étiquette est retenue si sa confiance dépasse le seuil
                # ADAPTATIF de SA PROPRE classe (et non le tau fixe global)
                threshold_per_sample = tau_per_class[pseudo_labels]
                mask = max_probs.ge(threshold_per_sample).float()

            # --- Étape 6 : perte de cohérence faible/forte, filtrée par le masque ---
            logits_u_s = model(imgs_u_s)
            loss_u_per_sample = F.cross_entropy(logits_u_s, pseudo_labels, reduction="none")
            loss_u = (loss_u_per_sample * mask).mean()

            # --- Étape 7 : perte totale ---
            loss = loss_s + cfg["lambda_u"] * loss_u

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
            "mask_rate": mask.mean().item(),  # fraction de pseudo-labels retenus -- utile à monitorer
        }

    return train_step
