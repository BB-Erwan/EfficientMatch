"""EfficientMatch (contribution de ce dépôt) -- FixMatch + un canal de Mixup FILTRÉ PAR LE MASQUE DE
CONFIANCE DUR, entre le batch labellisé et le batch non labellisé faiblement augmenté. Sur CIFAR-10.

Le canal FixMatch (perte supervisée + cohérence faible/forte seuillée) est identique à fixmatch.py.
Le canal Mixup ajoute : chaque échantillon (labellisé ou non, vue faible) est mélangé avec un
compagnon tiré au hasard dans l'union labellisé+non labellisé (même recette que mixmatch.py), mais la
perte de cette paire mixée est pondérée par le masque de confiance du COMPAGNON -- si le compagnon est
un pseudo-label rejeté (confiance < tau), la paire mixée est ignorée plutôt que d'entraîner le modèle
sur un mélange partiellement bruité.

Script autonome : tout ce qui concerne l'algorithme (augmentations, boucle, hyperparamètres) est ici.
Seuls le modèle (models.py), l'EMA (ema.py), l'évaluation top-1 (evaluate.py) et les utilitaires de
chargement de données génériques (data.py) sont partagés avec les autres algorithmes.

Usage :
    python efficientmatch.py
    python efficientmatch.py --n-labels 250 --K 65536 --lambda-mix 0.5
    python efficientmatch.py --verbose --debug-subset-size 2000 --K 200 --eval-every 50
"""
import os

# Contourne un bug Windows : le cache de compilation triton/inductor (utilisé par torch.compile,
# cf. plus bas) écrit par défaut dans %TEMP%\torchinductor_<user>\..., un chemin qui dépasse souvent
# la limite de 260 caractères de Windows une fois combiné aux sous-dossiers de hash de triton
# (FileNotFoundError silencieux à la toute première compilation). Un chemin court à la racine du
# disque système évite le problème -- DOIT être fait avant tout import de torch/triton, et seulement
# si l'utilisateur n'a pas déjà fixé ces variables lui-même.
if os.name == "nt":
    _cache_root = os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "tc")
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, "inductor"))
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Subset

from data import LabeledDataset, MultiViewDataset, infinite_loader, load_cifar10, make_ssl_split
from ema import EMA
from evaluate import evaluate
from models import WideResNet

# Racine du projet (parent de scripts/), calculée depuis l'emplacement de ce fichier -- PAS depuis le
# répertoire courant, pour toujours retomber sur le même dossier data/ quel que soit le cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Statistiques CIFAR-10 standard du protocole FixMatch/EfficientMatch.
MEAN, STD, IMAGE_SIZE, NUM_CLASSES = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616), 32, 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-labels", type=int, default=250)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--lambda-u", type=float, default=1.0, help="poids de la perte de cohérence FixMatch")
    parser.add_argument("--lambda-mix", type=float, default=1.0, help="poids du canal Mixup")
    parser.add_argument("--alpha-mix", type=float, default=0.75, help="paramètre de la loi Beta pour le Mixup")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                         help="calcule le forward/la perte en bfloat16 (torch.autocast) pour accélérer "
                              "l'entraînement -- pas de GradScaler nécessaire (bfloat16 a la même plage "
                              "d'exposant que fp32, pas de risque de sous-flottement des gradients)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                         help="torch.compile(model) pour accélérer l'entraînement (coût de compilation "
                              "ponctuel amorti sur toute la durée du run) -- ignoré silencieusement si "
                              "triton est indisponible")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen-factor", type=int, default=2)
    parser.add_argument("--K", type=int, default=2 ** 17)
    parser.add_argument("--eval-every", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="", help="étiquette libre incluse dans le nom du fichier de log")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--debug-subset-size", type=int, default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
                         help="affiche une ligne à CHAQUE itération (loss, it/s) pour suivre la vitesse en direct")
    parser.add_argument("--data-root", type=str, default=os.path.join(_PROJECT_ROOT, "data"))
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_schedule(optimizer, k, K, base_lr):
    """lr(k) = lr0 * cos(7*pi*k / (16*K)), schedule cosine recalé standard FixMatch/EfficientMatch."""
    new_lr = base_lr * math.cos(7 * math.pi * k / (16 * K))
    for group in optimizer.param_groups:
        group["lr"] = max(new_lr, 0.0)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Optimisations globales ──────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True       # sélectionne l'algo cuDNN le plus rapide
    torch.set_float32_matmul_precision("high")  # TF32 sur Ampere+ -- matmul plus rapide
    # ───────────────────────────────────────────────────────────────────────────

    # --- Augmentations : vue faible (flip+crop) et vue forte (+ RandAugment + RandomErasing), comme
    # FixMatch -- appliquées PAR ÉCHANTILLON (pas de vectorisation par batch) pour préserver la pleine
    # diversité d'augmentation. ---
    weak_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(IMAGE_SIZE, padding=4, padding_mode="reflect"),
        transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(MEAN, STD),
    ])
    strong_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(IMAGE_SIZE, padding=4, padding_mode="reflect"),
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.5),
    ])
    eval_transform = transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(MEAN, STD),
    ])

    # --- Données : split équilibré labellisé/non labellisé, deux vues (faible/forte) par échantillon
    # non labellisé ---
    train_base, test_base = load_cifar10(args.data_root)
    targets = np.array(train_base.targets)
    labeled_idx, unlabeled_idx = make_ssl_split(targets, args.n_labels, NUM_CLASSES, seed=args.seed)
    if args.debug_subset_size is not None:
        unlabeled_idx = unlabeled_idx[: args.debug_subset_size]
        test_base = Subset(test_base, list(range(min(len(test_base), args.debug_subset_size))))

    labeled_set = LabeledDataset(train_base, labeled_idx, weak_transform)
    unlabeled_set = MultiViewDataset(train_base, unlabeled_idx, [weak_transform, strong_transform])
    test_set = LabeledDataset(test_base, list(range(len(test_base))), eval_transform)

    labeled_iter = infinite_loader(labeled_set, args.B, args.num_workers, shuffle=True)
    unlabeled_iter = infinite_loader(unlabeled_set, args.mu * args.B, args.num_workers, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # --- Modèle + EMA débrayable (use_ema=False -> évalue directement les poids en cours d'entraînement) ---
    base_model = WideResNet(num_classes=NUM_CLASSES, depth=args.depth, widen_factor=args.widen_factor).to(device)
    model = base_model
    if args.compile:
        try:
            import triton  # noqa: F401
            # PAS mode="reduce-overhead" : ce mode active les CUDA Graphs, qui déclenchent sur Windows
            # un bug connu de torch._inductor (OverflowError: Python int too large to convert to C
            # long -- le C `long` Windows est 32 bits, contrairement à Linux). Le mode par défaut
            # (sans CUDA Graphs) compile et tourne normalement.
            model = torch.compile(base_model)
            print("torch.compile activé (mode par défaut -- reduce-overhead désactivé, bug Windows connu)")
        except ImportError:
            print("torch.compile demandé (--compile) mais triton indisponible -- modèle non compilé.")
    if args.use_ema:
        eval_model = WideResNet(num_classes=NUM_CLASSES, depth=args.depth, widen_factor=args.widen_factor).to(device)
        # EMA lit/écrit le state_dict de `base_model`, jamais celui de `model` (potentiellement
        # compilé) : torch.compile préfixe les clés du state_dict ("_orig_mod.xxx"), ce qui ferait
        # échouer eval_model.load_state_dict() plus bas (eval_model n'est jamais compilé).
        ema = EMA(base_model, args.ema_decay)
    else:
        eval_model = model
        ema = None
    optimizer = torch.optim.SGD(
        base_model.parameters(), lr=args.lr, momentum=args.momentum,
        nesterov=args.nesterov, weight_decay=args.weight_decay,
    )
    beta_dist = torch.distributions.Beta(
        torch.tensor(args.alpha_mix, device=device), torch.tensor(args.alpha_mix, device=device),
    )

    # --- Logging : même format que le reste du projet (config + logs + statut) pour rester
    # compatible avec analyze.py, quel que soit l'algorithme. ---
    cfg = {**vars(args), "algo": "efficientmatch", "dataset": "cifar10", "num_classes": NUM_CLASSES}
    suffix = f"_{args.tag}" if args.tag else ""
    log_path = f"./logs/efficientmatch_cifar10_n{args.n_labels}_K{args.K}_seed{args.seed}{suffix}.json"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Budget total : {args.K} itérations")

    logs = []
    mask_rate_since_eval = []
    pl_correct_since_eval, pl_retained_since_eval = 0, 0
    start_time = time.time()
    base_model.train()
    for k in range(1, args.K + 1):
        cosine_schedule(optimizer, k, args.K, args.lr)

        # --- batch labellisé ---
        imgs_x, labels_x = next(labeled_iter)
        imgs_x, labels_x = imgs_x.to(device), labels_x.to(device)

        # --- batch non labellisé, deux vues (faible / forte) + vrai label (chargé directement via le
        # DataLoader, avec le batch -- pas de lookup séparé -- pour le suivi diagnostique ci-dessous) ---
        imgs_u_w, imgs_u_s, true_u, _idx = next(unlabeled_iter)
        imgs_u_w, imgs_u_s, true_u = imgs_u_w.to(device), imgs_u_s.to(device), true_u.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            # --- pseudo-étiquetage sur la vue faible (sans gradient) + seuil fixe -- identique à
            # FixMatch, réutilisé aussi pour filtrer le canal Mixup ci-dessous ---
            with torch.no_grad():
                logits_u_w = model(imgs_u_w)
                probs_u_w = F.softmax(logits_u_w, dim=-1)
                max_probs, pseudo = probs_u_w.max(dim=-1)
                mask = max_probs.ge(args.tau).float()

            # --- canal FixMatch : perte supervisée + cohérence faible/forte filtrée par le masque ---
            logits_x = model(imgs_x)
            loss_s = F.cross_entropy(logits_x, labels_x)

            logits_u_s = model(imgs_u_s)
            loss_u_per_sample = F.cross_entropy(logits_u_s, pseudo, reduction="none")
            loss_u = (loss_u_per_sample * mask).mean()

            # --- canal Mixup : mélange chaque échantillon (labellisé ou non, vue faible) avec un
            # compagnon tiré au hasard dans l'union labellisé+non labellisé, lambda tiré d'une loi
            # Beta(alpha,alpha) replié en [0.5, 1] -- comme mixmatch.py -- puis la perte de chaque
            # paire mixée est pondérée par le masque de confiance du COMPAGNON (1 s'il est labellisé,
            # mask s'il est non labellisé) : une paire mélangée avec un pseudo-label rejeté est
            # ignorée plutôt que d'entraîner sur un mélange partiellement bruité. ---
            n_x = imgs_x.size(0)
            targets_x = F.one_hot(labels_x, NUM_CLASSES).float()
            targets_u_hard = F.one_hot(pseudo, NUM_CLASSES).float()

            pool_imgs = torch.cat([imgs_x, imgs_u_w], dim=0)
            pool_targets = torch.cat([targets_x, targets_u_hard], dim=0)
            pool_mask = torch.cat([torch.ones(n_x, device=device), mask], dim=0)
            shuffle_idx = torch.randperm(pool_imgs.size(0), device=device)
            companion_imgs = pool_imgs[shuffle_idx]
            companion_targets = pool_targets[shuffle_idx]
            companion_mask = pool_mask[shuffle_idx]

            lam_x = beta_dist.sample((n_x,))
            lam_u = beta_dist.sample((imgs_u_w.size(0),))
            lam_x = torch.maximum(lam_x, 1 - lam_x)
            lam_u = torch.maximum(lam_u, 1 - lam_u)
            lam_x_img, lam_u_img = lam_x.view(-1, 1, 1, 1), lam_u.view(-1, 1, 1, 1)
            lam_x_lbl, lam_u_lbl = lam_x.view(-1, 1), lam_u.view(-1, 1)

            mixed_x = lam_x_img * imgs_x + (1 - lam_x_img) * companion_imgs[:n_x]
            mixed_u = lam_u_img * imgs_u_w + (1 - lam_u_img) * companion_imgs[n_x:]
            mixed_targets_x = lam_x_lbl * targets_x + (1 - lam_x_lbl) * companion_targets[:n_x]
            mixed_targets_u = lam_u_lbl * probs_u_w + (1 - lam_u_lbl) * companion_targets[n_x:]

            logits_mixed = model(torch.cat([mixed_x, mixed_u], dim=0))
            logits_mixed_x, logits_mixed_u = logits_mixed[:n_x], logits_mixed[n_x:]
            loss_mixup_x = (companion_mask[:n_x] * F.cross_entropy(logits_mixed_x, mixed_targets_x, reduction="none")).mean()
            loss_mixup_u = (companion_mask[n_x:] * F.cross_entropy(logits_mixed_u, mixed_targets_u, reduction="none")).mean()
            loss_mixup = loss_mixup_x + loss_mixup_u

            loss = loss_s + args.lambda_u * loss_u + args.lambda_mix * loss_mixup
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(base_model)
        mask_rate_since_eval.append(mask.mean().item())

        # Qualité des pseudo-labels RETENUS (mask=1, donc effectivement utilisés dans le canal
        # FixMatch), comparés au vrai label -- diagnostic uniquement, jamais utilisé pour entraîner.
        retained = mask.bool()
        pl_correct_since_eval += (pseudo[retained] == true_u[retained]).sum().item()
        pl_retained_since_eval += retained.sum().item()

        is_eval_step = k % args.eval_every == 0 or k == args.K
        if is_eval_step:
            if ema is not None:
                ema.copy_to(eval_model)
            acc = evaluate(eval_model, test_loader, device)
            elapsed = time.time() - start_time
            # mask_rate moyenné depuis la dernière évaluation (pas la valeur instantanée de la
            # dernière itération) : plus représentatif de l'utilisation réelle des données non
            # labellisées sur la fenêtre, moins bruité d'une itération à l'autre.
            mean_mask_rate = sum(mask_rate_since_eval) / len(mask_rate_since_eval)
            mask_rate_since_eval = []
            pl_quality = pl_correct_since_eval / pl_retained_since_eval if pl_retained_since_eval > 0 else float("nan")
            pl_correct_since_eval, pl_retained_since_eval = 0, 0
            log_entry = {
                "iteration": k, "elapsed_seconds": elapsed, "eval_accuracy": acc,
                "loss": loss.item(), "loss_s": loss_s.item(), "loss_u": loss_u.item(),
                "loss_mixup": loss_mixup.item(), "mask_rate": mean_mask_rate, "pl_quality": pl_quality,
            }
            logs.append(log_entry)
            print(f"[iter {k:>7}/{args.K}] acc={acc:.4f} loss={loss.item():.4f} loss_s={loss_s.item():.3f} "
                  f"loss_u={loss_u.item():.3f} loss_mixup={loss_mixup.item():.3f} "
                  f"mask_rate={mean_mask_rate:.3f} pl_quality={pl_quality:.3f} elapsed={elapsed / 60:.1f}min")
            with open(log_path, "w") as f:
                json.dump({"config": cfg, "logs": logs}, f, indent=2)
        elif args.verbose:
            # Ligne écrasée en place (pas de retour à la ligne) : juste pour suivre la vitesse
            # d'exécution en direct, sans déclencher d'évaluation supplémentaire.
            it_per_sec = k / (time.time() - start_time)
            pl_q_now = (pseudo[retained] == true_u[retained]).float().mean().item() if retained.any() else float("nan")
            print(f"[iter {k:>7}/{args.K}] loss={loss.item():.4f} loss_s={loss_s.item():.3f} "
                  f"loss_u={loss_u.item():.3f} loss_mixup={loss_mixup.item():.3f} "
                  f"mask_rate={mask.mean().item():.3f} pl_quality={pl_q_now:.3f} ({it_per_sec:.2f} it/s)", end="\r")

    with open(log_path, "w") as f:
        json.dump({"config": cfg, "logs": logs, "status": "completed", "last_iteration": args.K}, f, indent=2)

    print("\nEntraînement terminé.")


if __name__ == "__main__":
    main()
