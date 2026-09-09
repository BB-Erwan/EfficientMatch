"""MixMatch (Berthelot et al., 2019) -- moyenne + sharpening de K_aug vues faibles pour le
pseudo-étiquetage non labellisé, puis Mixup entre labellisé et non labellisé, sur CIFAR-10.

Script autonome : tout ce qui concerne l'algorithme (augmentations, boucle, hyperparamètres) est ici.
Seuls le modèle (models.py), l'EMA (ema.py), l'évaluation top-1 (evaluate.py) et les utilitaires de
chargement de données génériques (data.py) sont partagés avec les autres algorithmes.

Usage :
    python mixmatch.py
    python mixmatch.py --alpha-mix 0.5 --K 65536
    python mixmatch.py --verbose --debug-subset-size 2000 --K 200 --eval-every 50
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

# Statistiques CIFAR-10 standard du protocole FixMatch/MixMatch.
MEAN, STD, IMAGE_SIZE, NUM_CLASSES = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616), 32, 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-labels", type=int, default=250)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--mu", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--K-aug", type=int, default=2, help="nombre de vues faibles moyennées pour le pseudo-étiquetage")
    parser.add_argument("--sharpen-T", type=float, default=0.5, help="température de sharpening")
    parser.add_argument("--alpha-mix", type=float, default=0.75, help="paramètre de la loi Beta pour le Mixup")
    parser.add_argument("--lambda-u-max", type=float, default=75.0,
                         help="poids max de la perte non supervisée (papier original CIFAR-10 : 75)")
    parser.add_argument("--rampup-length", type=int, default=16384,
                         help="nombre d'itérations pour le rampup linéaire de lambda_u")
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
    """lr(k) = lr0 * cos(7*pi*k / (16*K)), schedule cosine recalé standard FixMatch/MixMatch."""
    new_lr = base_lr * math.cos(7 * math.pi * k / (16 * K))
    for group in optimizer.param_groups:
        group["lr"] = max(new_lr, 0.0)


def estimate_flops_per_iter(args, device):
    """Mesure réelle des FLOPs (forward + backward) pour une itération MixMatch, sur un modèle
    jetable et des tenseurs factices de la bonne forme : K_aug forwards sans gradient (pseudo-
    étiquetage) + un forward avec gradient sur le batch mixé (labellisé + non labellisé confondus)."""
    from torch.utils.flop_counter import FlopCounterMode

    model = WideResNet(num_classes=NUM_CLASSES, depth=args.depth, widen_factor=args.widen_factor).to(device)
    model.train()
    B, muB = args.B, args.mu * args.B
    n_mixed = B + args.K_aug * muB
    dummy_u_views = [torch.randn(muB, 3, IMAGE_SIZE, IMAGE_SIZE, device=device) for _ in range(args.K_aug)]
    dummy_mixed = torch.randn(n_mixed, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    dummy_targets_x = torch.rand(B, NUM_CLASSES, device=device)
    dummy_targets_u = torch.rand(n_mixed - B, NUM_CLASSES, device=device)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as flop_counter:
        with torch.no_grad():
            for view in dummy_u_views:
                _ = model(view)
        logits = model(dummy_mixed)
        logits_x, logits_u = logits[:B], logits[B:]
        loss = (-(dummy_targets_x * F.log_softmax(logits_x, dim=-1)).sum(dim=-1).mean()
                + F.mse_loss(F.softmax(logits_u, dim=-1), dummy_targets_u))
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def estimate_time_per_iter(args, device, warmup_iters=20, bench_iters=100):
    """Chronomètre individuellement `bench_iters` itérations réelles (après `warmup_iters` itérations
    de chauffe non chronométrées) sur CIFAR-10 (respecte `args.debug_subset_size` pour rester rapide).
    Reproduit fidèlement la boucle de main(), sans EMA ni logging/évaluation.

    Retourne (moyenne, écart-type) en secondes/itération : un `bench_iters` pas trop petit et un
    écart-type explicite permettent de repérer une mesure encore instable (le tout début de
    l'entraînement peut être irrégulier) plutôt que de figer une moyenne trompeuse sur trop peu de
    points."""
    weak_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(IMAGE_SIZE, padding=4, padding_mode="reflect"),
        transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(MEAN, STD),
    ])
    train_base, _ = load_cifar10(args.data_root)
    targets = np.array(train_base.targets)
    labeled_idx, unlabeled_idx = make_ssl_split(targets, args.n_labels, NUM_CLASSES, seed=args.seed)
    if args.debug_subset_size is not None:
        unlabeled_idx = unlabeled_idx[: args.debug_subset_size]
    labeled_set = LabeledDataset(train_base, labeled_idx, weak_transform)
    unlabeled_set = MultiViewDataset(train_base, unlabeled_idx, [weak_transform] * args.K_aug)
    pin_memory = args.pin_memory and device.type == "cuda"
    labeled_iter = infinite_loader(labeled_set, args.B, args.num_workers, shuffle=True, pin_memory=pin_memory)
    unlabeled_iter = infinite_loader(
        unlabeled_set, args.mu * args.B, args.num_workers, shuffle=True, pin_memory=pin_memory,
    )

    base_model = WideResNet(num_classes=NUM_CLASSES, depth=args.depth, widen_factor=args.widen_factor).to(device)
    model = base_model
    if args.compile:
        try:
            import triton  # noqa: F401
            model = torch.compile(base_model)
        except ImportError:
            pass
    optimizer = torch.optim.SGD(
        base_model.parameters(), lr=args.lr, momentum=args.momentum,
        nesterov=args.nesterov, weight_decay=args.weight_decay,
    )
    beta_dist = torch.distributions.Beta(
        torch.tensor(args.alpha_mix, device=device), torch.tensor(args.alpha_mix, device=device),
    )
    base_model.train()

    def step(k):
        cosine_schedule(optimizer, k, args.K, args.lr)
        imgs_x, labels_x = next(labeled_iter)
        imgs_x, labels_x = imgs_x.to(device, non_blocking=True), labels_x.to(device, non_blocking=True)
        targets_x = F.one_hot(labels_x, NUM_CLASSES).float()
        *imgs_u_views, _true_u, _idx = next(unlabeled_iter)
        imgs_u_views = [view.to(device, non_blocking=True) for view in imgs_u_views]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            with torch.no_grad():
                probs_u = torch.stack([F.softmax(model(v), dim=-1) for v in imgs_u_views], dim=0).mean(dim=0)
                targets_u = probs_u.pow(1.0 / args.sharpen_T)
                targets_u = targets_u / targets_u.sum(dim=-1, keepdim=True)
            n_x = imgs_x.size(0)
            all_imgs = torch.cat([imgs_x] + imgs_u_views, dim=0)
            all_targets = torch.cat([targets_x] + [targets_u] * len(imgs_u_views), dim=0)
            shuffle_idx = torch.randperm(all_imgs.size(0), device=device)
            companion_imgs, companion_targets = all_imgs[shuffle_idx], all_targets[shuffle_idx]
            lam = beta_dist.sample((all_imgs.size(0),))
            lam = torch.maximum(lam, 1 - lam)
            lam_imgs, lam_targets = lam.view(-1, 1, 1, 1), lam.view(-1, 1)
            mixed_imgs = lam_imgs * all_imgs + (1 - lam_imgs) * companion_imgs
            mixed_targets = lam_targets * all_targets + (1 - lam_targets) * companion_targets
            logits = model(mixed_imgs)
            logits_x, logits_u = logits[:n_x], logits[n_x:]
            loss_s = -(mixed_targets[:n_x] * F.log_softmax(logits_x, dim=-1)).sum(dim=-1).mean()
            loss_u = F.mse_loss(F.softmax(logits_u, dim=-1), mixed_targets[n_x:])
            lambda_u = args.lambda_u_max * min(1.0, k / args.rampup_length)
            loss = loss_s + lambda_u * loss_u
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

    # --- Augmentations : MixMatch n'utilise que des vues FAIBLES (flip+crop), K_aug vues
    # indépendantes de la même image, moyennées puis sharpened pour le pseudo-étiquetage -- pas de
    # RandAugment/vue forte contrairement à FixMatch. Appliquées PAR ÉCHANTILLON (pas de
    # vectorisation par batch) pour préserver la pleine diversité d'augmentation. ---
    weak_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(IMAGE_SIZE, padding=4, padding_mode="reflect"),
        transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(MEAN, STD),
    ])

    # --- Données : split équilibré labellisé/non labellisé, K_aug vues faibles par échantillon non
    # labellisé ---
    train_base, test_base = load_cifar10(args.data_root)
    targets = np.array(train_base.targets)
    labeled_idx, unlabeled_idx = make_ssl_split(targets, args.n_labels, NUM_CLASSES, seed=args.seed)
    if args.debug_subset_size is not None:
        unlabeled_idx = unlabeled_idx[: args.debug_subset_size]
        test_base = Subset(test_base, list(range(min(len(test_base), args.debug_subset_size))))

    labeled_set = LabeledDataset(train_base, labeled_idx, weak_transform)
    unlabeled_set = MultiViewDataset(train_base, unlabeled_idx, [weak_transform] * args.K_aug)
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
    cfg = {**vars(args), "algo": "mixmatch", "dataset": "cifar10", "num_classes": NUM_CLASSES}
    suffix = f"_{args.tag}" if args.tag else ""
    log_path = f"./logs/mixmatch_cifar10_n{args.n_labels}_K{args.K}_seed{args.seed}{suffix}.json"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Budget total : {args.K} itérations")

    logs = []
    pl_correct_since_eval, pl_total_since_eval = 0, 0
    start_time = time.time()
    base_model.train()
    for k in range(1, args.K + 1):
        cosine_schedule(optimizer, k, args.K, args.lr)

        # --- batch labellisé ---
        imgs_x, labels_x = next(labeled_iter)
        imgs_x, labels_x = imgs_x.to(device, non_blocking=True), labels_x.to(device, non_blocking=True)
        targets_x = F.one_hot(labels_x, NUM_CLASSES).float()

        # --- K_aug vues faibles du batch non labellisé + vrai label (chargé directement via le
        # DataLoader, avec le batch -- pas de lookup séparé -- pour le suivi diagnostique ci-dessous) ---
        *imgs_u_views, true_u, _idx = next(unlabeled_iter)
        imgs_u_views = [view.to(device, non_blocking=True) for view in imgs_u_views]
        true_u = true_u.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            # --- pseudo-étiquetage par moyenne des K_aug vues + sharpening (sans gradient) ---
            with torch.no_grad():
                probs_u = torch.stack([F.softmax(model(view), dim=-1) for view in imgs_u_views], dim=0).mean(dim=0)
                targets_u = probs_u.pow(1.0 / args.sharpen_T)
                targets_u = targets_u / targets_u.sum(dim=-1, keepdim=True)

            # --- Mixup : mélange chaque échantillon (labellisé ou non) avec un compagnon tiré au
            # hasard dans l'union labellisé+non labellisé, lambda tiré d'une loi Beta(alpha,alpha) et
            # replié en [0.5, 1] pour que le mélange reste proche de l'échantillon d'origine ---
            n_x = imgs_x.size(0)
            all_imgs = torch.cat([imgs_x] + imgs_u_views, dim=0)
            all_targets = torch.cat([targets_x] + [targets_u] * len(imgs_u_views), dim=0)
            shuffle_idx = torch.randperm(all_imgs.size(0), device=device)
            companion_imgs, companion_targets = all_imgs[shuffle_idx], all_targets[shuffle_idx]

            lam = beta_dist.sample((all_imgs.size(0),))
            lam = torch.maximum(lam, 1 - lam)
            lam_imgs = lam.view(-1, 1, 1, 1)
            lam_targets = lam.view(-1, 1)

            mixed_imgs = lam_imgs * all_imgs + (1 - lam_imgs) * companion_imgs
            mixed_targets = lam_targets * all_targets + (1 - lam_targets) * companion_targets

            # --- un seul forward sur le batch mixé (labellisé + non labellisé confondus, comme au
            # Mixup) -- puis pertes séparées : cross-entropy douce sur la partie labellisée, MSE sur
            # la partie non labellisée (papier MixMatch, pas de seuil de confiance contrairement à
            # FixMatch) ---
            logits = model(mixed_imgs)
            logits_x, logits_u = logits[:n_x], logits[n_x:]

            loss_s = -(mixed_targets[:n_x] * F.log_softmax(logits_x, dim=-1)).sum(dim=-1).mean()
            loss_u = F.mse_loss(F.softmax(logits_u, dim=-1), mixed_targets[n_x:])

            # --- rampup linéaire du poids de la perte non supervisée ---
            lambda_u = args.lambda_u_max * min(1.0, k / args.rampup_length)
            loss = loss_s + lambda_u * loss_u
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(base_model)

        # Qualité des pseudo-labels : MixMatch n'a pas de seuil de confiance (pas de notion
        # d'échantillon "retenu" vs rejeté, contrairement à FixMatch/FlexMatch/EfficientMatch) --
        # comparé au vrai label, le pseudo-label ici est simplement l'argmax de la cible sharpened
        # (moyenne des K_aug vues + sharpening). Diagnostic uniquement, jamais utilisé pour entraîner.
        pseudo_u = targets_u.argmax(dim=-1)
        pl_correct_since_eval += (pseudo_u == true_u).sum().item()
        pl_total_since_eval += true_u.size(0)

        is_eval_step = k % args.eval_every == 0 or k == args.K
        if is_eval_step:
            if ema is not None:
                ema.copy_to(eval_model)
            acc = evaluate(eval_model, test_loader, device)
            elapsed = time.time() - start_time
            pl_quality = pl_correct_since_eval / pl_total_since_eval if pl_total_since_eval > 0 else float("nan")
            pl_correct_since_eval, pl_total_since_eval = 0, 0
            log_entry = {
                "iteration": k, "elapsed_seconds": elapsed, "eval_accuracy": acc,
                "loss": loss.item(), "loss_s": loss_s.item(), "loss_u": loss_u.item(), "lambda_u": lambda_u,
                "pl_quality": pl_quality,
            }
            logs.append(log_entry)
            print(f"[iter {k:>7}/{args.K}] acc={acc:.4f} loss={loss.item():.4f} loss_s={loss_s.item():.3f} "
                  f"loss_u={loss_u.item():.3f} lambda_u={lambda_u:.3f} pl_quality={pl_quality:.3f} "
                  f"elapsed={elapsed / 60:.1f}min")
            with open(log_path, "w") as f:
                json.dump({"config": cfg, "logs": logs}, f, indent=2)
        elif args.verbose:
            # Ligne écrasée en place (pas de retour à la ligne) : juste pour suivre la vitesse
            # d'exécution en direct, sans déclencher d'évaluation supplémentaire.
            it_per_sec = k / (time.time() - start_time)
            pl_q_now = (pseudo_u == true_u).float().mean().item()
            print(f"[iter {k:>7}/{args.K}] loss={loss.item():.4f} loss_s={loss_s.item():.3f} "
                  f"loss_u={loss_u.item():.3f} lambda_u={lambda_u:.3f} pl_quality={pl_q_now:.3f} "
                  f"({it_per_sec:.2f} it/s)", end="\r")

    with open(log_path, "w") as f:
        json.dump({"config": cfg, "logs": logs, "status": "completed", "last_iteration": args.K}, f, indent=2)

    print("\nEntraînement terminé.")


if __name__ == "__main__":
    main()
