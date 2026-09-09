"""FixMatch (Sohn et al., 2020) -- perte supervisée + cohérence faible/forte seuillée."""
import torch
import torch.nn.functional as F


def make_train_step(cfg, device):
    def train_step(model, optimizer, labeled_iter, unlabeled_iter):
        # --- Étape 1 : batch labellisé ---
        imgs_x, labels_x = next(labeled_iter)
        imgs_x, labels_x = imgs_x.to(device), labels_x.to(device)

        # --- Étape 2 : batch non labellisé, deux vues (faible / forte) ---
        imgs_u_w, imgs_u_s = next(unlabeled_iter)
        imgs_u_w, imgs_u_s = imgs_u_w.to(device), imgs_u_s.to(device)

        optimizer.zero_grad(set_to_none=True)

        # --- Étape 3 : perte supervisée ---
        logits_x = model(imgs_x)
        loss_s = F.cross_entropy(logits_x, labels_x)

        # --- Étape 4 : pseudo-étiquetage sur la vue faible (sans gradient) ---
        with torch.no_grad():
            logits_u_w = model(imgs_u_w)
            probs_u_w = F.softmax(logits_u_w, dim=-1)
            max_probs, pseudo_labels = probs_u_w.max(dim=-1)

        # --- Étape 5 : seuil fixe ---
        mask = max_probs.ge(cfg["tau"]).float()

        # --- Étape 6 : perte de cohérence faible/forte, filtrée par le masque ---
        logits_u_s = model(imgs_u_s)
        loss_u_per_sample = F.cross_entropy(logits_u_s, pseudo_labels, reduction="none")
        loss_u = (loss_u_per_sample * mask).mean()

        # --- Étape 7 : perte totale ---
        loss = loss_s + cfg["lambda_u"] * loss_u

        # --- Étape 8 : backward + optimisation ---
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(), "loss_s": loss_s.item(), "loss_u": loss_u.item(),
            "mask_rate": mask.mean().item(),
        }

    return train_step
