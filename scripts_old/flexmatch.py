"""FlexMatch (Zhang et al., 2021) -- FixMatch + Curriculum Pseudo Labeling (CPL) : seuil de
confiance ADAPTATIF par classe (au lieu d'un seuil fixe `tau`), pour corriger le biais envers les
classes "faciles" que le modèle apprend plus vite. Sur CIFAR-10.

Script autonome : tout ce qui concerne l'algorithme (augmentations, boucle, hyperparamètres) est ici.
Seuls le modèle (models.py), l'EMA (ema.py), l'évaluation top-1 (evaluate.py) et les utilitaires de
chargement de données génériques (data.py) sont partagés avec les autres algorithmes.

Usage :
    python flexmatch.py
    python flexmatch.py --n-labels 250 --K 65536 --tau 0.9
    python flexmatch.py --verbose --debug-subset-size 2000 --K 200 --eval-every 50
"""
import os

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

# Statistiques CIFAR-10 standard du protocole FixMatch/FlexMatch.
MEAN, STD, IMAGE_SIZE, NUM_CLASSES = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616), 32, 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-labels", type=int, default=250)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--mu", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                         help="calcule le forward/la perte en bfloat16 (torch.autocast) pour accélérer "
                              "l'entraînement -- pas de GradScaler nécessaire (bfloat16 a la même plage "
                              "d'exposant que fp32, pas de risque de sous-flottement des gradients)")
    
    parser.add_argument("--thresh-warmup", action=argparse.BooleanOptionalAction, default=True,
                         help="normalise l'effet d'apprentissage par classe en comptant aussi les "
                              "échantillons jamais encore sélectionnés (accélère la montée en confiance "
                              "en tout début d'entraînement, cf. papier FlexMatch)")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen-factor", type=int, default=2)
    parser.add_argument("--K", type=int, default=2 ** 17)
    parser.add_argument("--eval-every", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="", help="étiquette libre incluse dans le nom du fichier de log")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True,
                         help="epingle les batches en memoire page-verrouillee (DataLoader "
                              "pin_memory) pour un transfert CPU->GPU asynchrone plus rapide "
                              "(non_blocking=True) -- sans effet sur CPU")
    parser.add_argument("--debug-subset-size", type=int, default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
                         help="affiche une ligne à CHAQUE itération (loss, it/s) pour suivre la vitesse en direct")
    parser.add_argument("--data-root", type=str, default=os.path.join(_PROJECT_ROOT, "data"))
    return parser.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_schedule(optimizer, k, K, base_lr):
    """lr(k) = lr0 * cos(7*pi*k / (16*K)), schedule cosine recalé standard FixMatch/FlexMatch."""
    new_lr = base_lr * math.cos(7 * math.pi * k / (16 * K))
    for group in optimizer.param_groups:
        group["lr"] = max(new_lr, 0.0)


def estimate_flops_per_iter(args, device):
    """Mesure réelle des FLOPs (forward + backward) pour une itération FlexMatch, sur un modèle
    jetable et des tenseurs factices de la bonne forme. Identique à FixMatch côté FLOPs : le CPL
    (seuil adaptatif, bookkeeping `classwise_acc`) n'ajoute aucun forward réseau supplémentaire,
    juste de l'indexation/comptage sur des tenseurs 1D (coût négligeable, non mesurable ici)."""
    from torch.utils.flop_counter import FlopCounterMode

    model = WideResNet(num_classes=NUM_CLASSES, depth=args.depth, widen_factor=args.widen_factor).to(device)
    model.train()
    B, muB = args.B, args.mu * args.B
    dummy_x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    dummy_u_w = torch.randn(muB, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    dummy_u_s = torch.randn(muB, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    dummy_labels_x = torch.randint(0, NUM_CLASSES, (B,), device=device)
    dummy_labels_u = torch.randint(0, NUM_CLASSES, (muB,), device=device)
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


def estimate_time_per_iter(args, device, warmup_iters=20, bench_iters=100):
    """Chronomètre individuellement `bench_iters` itérations réelles (après `warmup_iters` itérations
    de chauffe non chronométrées) sur CIFAR-10 (respecte `args.debug_subset_size` pour rester rapide).
    Reproduit fidèlement la boucle de main() (y compris le bookkeeping CPL), sans EMA ni
    logging/évaluation.

    Retourne (moyenne, écart-type) en secondes/itération : un `bench_iters` pas trop petit et un
    écart-type explicite permettent de repérer une mesure encore instable (le tout début de
    l'entraînement peut être irrégulier) plutôt que de figer une moyenne trompeuse sur trop peu de
    points."""
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
    train_base, _ = load_cifar10(args.data_root)
    targets = np.array(train_base.targets)
    labeled_idx, unlabeled_idx = make_ssl_split(targets, args.n_labels, NUM_CLASSES, seed=args.seed)
    if args.debug_subset_size is not None:
        unlabeled_idx = unlabeled_idx[: args.debug_subset_size]
    labeled_set = LabeledDataset(train_base, labeled_idx, weak_transform)
    unlabeled_set = MultiViewDataset(train_base, unlabeled_idx, [weak_transform, strong_transform])
    pin_memory = args.pin_memory and device.type == "cuda"
    labeled_iter = infinite_loader(labeled_set, args.B, args.num_workers, shuffle=True, pin_memory=pin_memory)
    unlabeled_iter = infinite_loader(
        unlabeled_set, args.mu * args.B, args.num_workers, shuffle=True, pin_memory=pin_memory,
    )

    model = WideResNet(num_classes=NUM_CLASSES, depth=args.depth, widen_factor=args.widen_factor).to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        nesterov=args.nesterov, weight_decay=args.weight_decay,
    )
    model.train()
    selected_label = torch.full((len(unlabeled_idx),), -1, dtype=torch.long, device=device)
    classwise_acc = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)

    def step(k):
        nonlocal classwise_acc
        cosine_schedule(optimizer, k, args.K, args.lr)
        imgs_x, labels_x = next(labeled_iter)
        imgs_x, labels_x = imgs_x.to(device, non_blocking=True), labels_x.to(device, non_blocking=True)
        imgs_u_w, imgs_u_s, _true_u, idx = next(unlabeled_iter)
        imgs_u_w, imgs_u_s, idx = (
            imgs_u_w.to(device, non_blocking=True), imgs_u_s.to(device, non_blocking=True),
            idx.to(device, non_blocking=True),
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            with torch.no_grad():
                logits_u_w = model(imgs_u_w)
                max_probs, pseudo = F.softmax(logits_u_w, dim=-1).max(dim=-1)
                acc_per_sample = classwise_acc[pseudo]
                flexible_thresh = args.tau * (acc_per_sample / (2.0 - acc_per_sample))
                mask = max_probs.ge(flexible_thresh).float()
                select = max_probs.ge(args.tau)
                if select.any():
                    selected_label[idx[select]] = pseudo[select]
                counts = torch.bincount(selected_label[selected_label != -1] + 1, minlength=NUM_CLASSES + 1)
                if counts.max().item() < selected_label.shape[0]:
                    counts_per_class = counts[1:].float()
                    denom = max(counts.max().item(), 1) if args.thresh_warmup else max(
                        (counts.clone().index_fill_(0, torch.tensor([0], device=device), 0)).max().item(), 1)
                    classwise_acc = counts_per_class / denom
            logits_x = model(imgs_x)
            loss_s = F.cross_entropy(logits_x, labels_x)
            logits_u_s = model(imgs_u_s)
            loss_u = (F.cross_entropy(logits_u_s, pseudo, reduction="none") * mask).mean()
            loss = loss_s + args.lambda_u * loss_u
        loss.backward()
        optimizer.step()

    for k in range(1, warmup_iters + 1):
        step(k)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for k in range(warmup_iters + 1, warmup_iters + bench_iters + 1):
        t0 = time.time()
        step(k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.time() - t0)

    timings = torch.tensor(timings)
    return timings.mean().item(), timings.std().item()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Optimisations globales ──────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True       # sélectionne l'algo cuDNN le plus rapide
    torch.set_float32_matmul_precision("high")  # TF32 sur Ampere+ -- matmul plus rapide
    # ───────────────────────────────────────────────────────────────────────────

    # --- Augmentations : vue faible (flip+crop) et vue forte (+ RandAugment + RandomErasing),
    # appliquées PAR ÉCHANTILLON (pas de vectorisation par batch) pour préserver la pleine diversité
    # d'augmentation -- cf. mémoire projet, un compromis vitesse/précision déjà tranché en faveur de
    # la précision. ---
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

    # --- Données : split équilibré labellisé/non labellisé, deux vues (faible/forte) + indice par
    # échantillon non labellisé (indice nécessaire au CPL, cf. MultiViewDataset) ---
    train_base, test_base = load_cifar10(args.data_root)
    targets = np.array(train_base.targets)
    labeled_idx, unlabeled_idx = make_ssl_split(targets, args.n_labels, NUM_CLASSES, seed=args.seed)
    if args.debug_subset_size is not None:
        unlabeled_idx = unlabeled_idx[: args.debug_subset_size]
        test_base = Subset(test_base, list(range(min(len(test_base), args.debug_subset_size))))

    labeled_set = LabeledDataset(train_base, labeled_idx, weak_transform)
    unlabeled_set = MultiViewDataset(train_base, unlabeled_idx, [weak_transform, strong_transform])
    test_set = LabeledDataset(test_base, list(range(len(test_base))), eval_transform)

    pin_memory = args.pin_memory and device.type == "cuda"
    labeled_iter = infinite_loader(labeled_set, args.B, args.num_workers, shuffle=True, pin_memory=pin_memory)
    unlabeled_iter = infinite_loader(
        unlabeled_set, args.mu * args.B, args.num_workers, shuffle=True, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, persistent_workers=args.num_workers > 0,
    )

    # --- Modèle + EMA débrayable (use_ema=False -> évalue directement les poids en cours d'entraînement) ---
    model = WideResNet(num_classes=NUM_CLASSES, depth=args.depth, widen_factor=args.widen_factor).to(device)
    if args.use_ema:
        eval_model = WideResNet(num_classes=NUM_CLASSES, depth=args.depth, widen_factor=args.widen_factor).to(device)
        ema = EMA(model, args.ema_decay)
    else:
        eval_model = model
        ema = None
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        nesterov=args.nesterov, weight_decay=args.weight_decay,
    )

    # --- État du Curriculum Pseudo Labeling (CPL), maintenu à travers les itérations :
    # `selected_label` : dernière classe assignée à chaque échantillon non labellisé quand sa
    #   confiance a dépassé le seuil FIXE `tau` (indépendant du seuil flexible utilisé pour la perte).
    # `classwise_acc` : effet d'apprentissage estimé par classe (sigma_t(c) du papier) -- proportion
    #   d'échantillons actuellement sélectionnés dans la classe la plus représentée, dans [0, 1].
    #   Sert à moduler le seuil de confiance par classe (cf. boucle) : proche de 0 pour une classe
    #   encore mal apprise (peu d'échantillons sélectionnés -> seuil bas, on accepte plus de
    #   pseudo-labels de cette classe), proche de 1 pour une classe déjà bien apprise (seuil ~= tau). ---
    selected_label = torch.full((len(unlabeled_idx),), -1, dtype=torch.long, device=device)
    classwise_acc = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)

    # --- Logging : même format que le reste du projet (config + logs + statut) pour rester
    # compatible avec analyze.py, quel que soit l'algorithme. ---
    cfg = {**vars(args), "algo": "flexmatch", "dataset": "cifar10", "num_classes": NUM_CLASSES}
    suffix = f"_{args.tag}" if args.tag else ""
    log_path = f"./logs/flexmatch_cifar10_n{args.n_labels}_K{args.K}_seed{args.seed}{suffix}.json"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Budget total : {args.K} itérations")

    logs = []
    mask_rate_since_eval = []
    pl_correct_since_eval, pl_retained_since_eval = 0, 0
    start_time = time.time()
    model.train()
    for k in range(1, args.K + 1):
        cosine_schedule(optimizer, k, args.K, args.lr)

        # --- batch labellisé ---
        imgs_x, labels_x = next(labeled_iter)
        imgs_x, labels_x = imgs_x.to(device, non_blocking=True), labels_x.to(device, non_blocking=True)

        # --- batch non labellisé, deux vues (faible / forte) + vrai label (chargé directement via le
        # DataLoader, avec le batch -- pas de lookup séparé -- pour le suivi diagnostique ci-dessous) +
        # indice (nécessaire au CPL, cf. `selected_label` plus bas) ---
        imgs_u_w, imgs_u_s, true_u, idx = next(unlabeled_iter)
        imgs_u_w, imgs_u_s, true_u, idx = (
            imgs_u_w.to(device, non_blocking=True), imgs_u_s.to(device, non_blocking=True),
            true_u.to(device, non_blocking=True), idx.to(device, non_blocking=True),
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            # --- pseudo-étiquetage sur la vue faible (sans gradient) + seuil ADAPTATIF par classe ---
            with torch.no_grad():
                logits_u_w = model(imgs_u_w)
                probs_u_w = F.softmax(logits_u_w, dim=-1)
                max_probs, pseudo = probs_u_w.max(dim=-1)

                # Seuil flexible (papier, §CPL) : mapping convexe beta/(2-beta) de l'effet
                # d'apprentissage de la classe prédite, qui module le tau fixe -- utilisé pour la
                # perte de cohérence.
                acc_per_sample = classwise_acc[pseudo]
                flexible_thresh = args.tau * (acc_per_sample / (2.0 - acc_per_sample))
                mask = max_probs.ge(flexible_thresh).float()

                # Bookkeeping pour la MAJ de classwise_acc à l'itération suivante -- seuil FIXE `tau`
                # (indépendant du seuil flexible ci-dessus), cf. papier.
                select = max_probs.ge(args.tau)
                if select.any():
                    selected_label[idx[select]] = pseudo[select]
                counts = torch.bincount(selected_label[selected_label != -1] + 1, minlength=NUM_CLASSES + 1)
                if counts.max().item() < selected_label.shape[0]:
                    counts_per_class = counts[1:].float()
                    if args.thresh_warmup:
                        denom = max(counts.max().item(), 1)
                    else:
                        wo_negative_one = counts.clone()
                        wo_negative_one[0] = 0
                        denom = max(wo_negative_one.max().item(), 1)
                    classwise_acc = counts_per_class / denom

            # --- perte supervisée ---
            logits_x = model(imgs_x)
            loss_s = F.cross_entropy(logits_x, labels_x)

            # --- perte de cohérence faible/forte, filtrée par le masque flexible ---
            logits_u_s = model(imgs_u_s)
            loss_u_per_sample = F.cross_entropy(logits_u_s, pseudo, reduction="none")
            loss_u = (loss_u_per_sample * mask).mean()

            loss = loss_s + args.lambda_u * loss_u
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(model)
        mask_rate_since_eval.append(mask.mean().item())

        # Qualité des pseudo-labels RETENUS (mask=1, donc effectivement utilisés dans la perte de
        # cohérence), comparés au vrai label -- diagnostic uniquement, jamais utilisé pour entraîner.
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
                "mask_rate": mean_mask_rate, "pl_quality": pl_quality,
            }
            logs.append(log_entry)
            print(f"[iter {k:>7}/{args.K}] acc={acc:.4f} loss={loss.item():.4f} loss_s={loss_s.item():.3f} "
                  f"loss_u={loss_u.item():.3f} mask_rate={mean_mask_rate:.3f} pl_quality={pl_quality:.3f} "
                  f"elapsed={elapsed / 60:.1f}min")
            with open(log_path, "w") as f:
                json.dump({"config": cfg, "logs": logs}, f, indent=2)
        elif args.verbose:
            # Ligne écrasée en place (pas de retour à la ligne) : juste pour suivre la vitesse
            # d'exécution en direct, sans déclencher d'évaluation supplémentaire.
            it_per_sec = k / (time.time() - start_time)
            pl_q_now = (pseudo[retained] == true_u[retained]).float().mean().item() if retained.any() else float("nan")
            print(f"[iter {k:>7}/{args.K}] loss={loss.item():.4f} loss_s={loss_s.item():.3f} "
                  f"loss_u={loss_u.item():.3f} mask_rate={mask.mean().item():.3f} pl_quality={pl_q_now:.3f} "
                  f"({it_per_sec:.2f} it/s)", end="\r")

    with open(log_path, "w") as f:
        json.dump({"config": cfg, "logs": logs, "status": "completed", "last_iteration": args.K}, f, indent=2)

    print("\nEntraînement terminé.")


if __name__ == "__main__":
    main()
