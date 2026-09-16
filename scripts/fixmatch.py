import argparse
import json
import logging
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

sys.path.append("..")  # add parent directory to path for imports

from datasets_utils import TransformedDataset, TransformedDatasetWithIndex
from models import WideResNet, build_model
from utils import evaluate_f1_and_accuracy, build_lr_scheduler
from ema import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_pseudo_label_metrics(pseudo_labels, confidences, last_pseudo_labels, last_confidences, true_labels):
    """Track how pseudo-labels evolve between two evaluations: new labels assigned,
    corrections (wrong -> right) vs bad corrections (right -> wrong), and whether
    confidence increases reinforce correct or incorrect labels."""
    new_label = (pseudo_labels != last_pseudo_labels) & (last_pseudo_labels == -1)
    existing_label_changes = (pseudo_labels != last_pseudo_labels) & (last_pseudo_labels != -1)
    was_good_pseudo = last_pseudo_labels == true_labels
    is_good_pseudo = pseudo_labels == true_labels
    confidence_increased = confidences > last_confidences

    return {
        "pl_quality": is_good_pseudo.sum().item() / len(true_labels),
        "corrections": (existing_label_changes & ~was_good_pseudo & is_good_pseudo).sum().item(),
        "bad_corrections": (existing_label_changes & was_good_pseudo & ~is_good_pseudo).sum().item(),
        "new_label": new_label.sum().item(),
        "new_errors": (new_label & ~is_good_pseudo).sum().item(),
        "new_correct": (new_label & is_good_pseudo).sum().item(),
        "correct_reinforcement": (confidence_increased & is_good_pseudo & was_good_pseudo & ~new_label).sum().item(),
        "error_reinforcement": (confidence_increased & ~is_good_pseudo & ~was_good_pseudo & ~new_label).sum().item(),
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
parser.add_argument("--widen_factor", type=int, default=2, help="WideResNet-28-{widen_factor}. Papers use 2 for CIFAR-10 and 8 for CIFAR-100.")
parser.add_argument("--model", type=str, default="wideresnet", choices=["wideresnet", "resnet18"], help="Backbone architecture.")
parser.add_argument("--depth", type=int, default=28, help="WideResNet depth (e.g. 28 for WRN-28-x, 40 for WRN-40-x). Ignored for resnet18. Must satisfy (depth-4) mod 6 == 0.")
parser.add_argument("--topk", type=int, default=1, help="Also report top-k accuracy for every k from 1 to this value (e.g. --topk 3 logs Top-1, Top-2 and Top-3).")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="SGD weight decay. Papers use 5e-4 for CIFAR-10 and 1e-3 for CIFAR-100.")
parser.add_argument("--num_labeled", type=int, default=250)
parser.add_argument("--optimized", type=str2bool, default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test_period", type=int, default=500)
parser.add_argument("--tau", type=float, default=0.95)
parser.add_argument("--max_steps", type=int, default=2**20, help="Number of steps actually run; the run is truncated here.")
parser.add_argument("--total_steps", type=int, default=2**20, help="Nominal horizon the cosine LR schedule decays over, independent of max_steps.")
parser.add_argument("--lr_schedule", type=str, default="fixmatch_cosine", choices=["fixmatch_cosine", "cosine_annealing"], help="LR schedule: rescaled FixMatch cosine (default) or torch's classic CosineAnnealingLR.")
parser.add_argument("--verbose", type=str2bool, default=False)
parser.add_argument("--target_acc", type=float, default=None, help="Stop the run early once test_acc reaches this value.")
parser.add_argument("--use_ema", type=str2bool, default=True, help="Evaluate an EMA of the weights instead of the raw training weights.")
parser.add_argument("--ema_decay", type=float, default=0.999)
args = parser.parse_args()


def run_fixmatch():
    # ── Optimisations globales ──────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True  # Sélectionne l'algo cuDNN le plus rapide
    torch.set_float32_matmul_precision("high")  # TF32 sur Ampere (A4000) — matmul plus rapide
    # ───────────────────────────────────────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "cifar100":
        num_classes = 100
        mean = torch.tensor([0.5071, 0.4865, 0.4409])
        std = torch.tensor([0.2673, 0.2564, 0.2762])
        train_ds = torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
        test_ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
    elif args.dataset == "svhn":
        num_classes = 10
        mean = torch.tensor([0.4377, 0.4438, 0.4728])
        std = torch.tensor([0.1980, 0.2010, 0.1970])
        train_ds = torchvision.datasets.SVHN(root="./data", split="train", download=True)
        test_ds = torchvision.datasets.SVHN(root="./data", split="test", download=True)
    else:
        num_classes = 10
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])
        train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

    logger.info(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"Mean: {mean}, Std: {std}")

    num_labeled = args.num_labeled
    optimized = args.optimized
    torch.manual_seed(args.seed)
    # do a randperm on the training dataset to shuffle it
    train_ds = Subset(train_ds, torch.randperm(len(train_ds)))

    # Split the training data into labeled and unlabeled datasets with balanced classes
    class_counts = torch.zeros(num_classes)
    for _, label in train_ds:
        class_counts[label] += 1

    num_per_class = num_labeled // num_classes
    labeled_indices = []
    unlabeled_indices = []

    for i in range(num_classes):
        class_indices = [j for j, (_, label) in enumerate(train_ds) if label == i]
        perm = torch.randperm(len(class_indices))
        labeled_indices.extend([class_indices[j] for j in perm[:num_per_class]])
        unlabeled_indices.extend([class_indices[j] for j in perm[num_per_class:]])

    labeled_ds = Subset(train_ds, labeled_indices)
    unlabeled_ds = Subset(train_ds, unlabeled_indices)

    # check class distribution in labeled dataset
    labeled_class_counts = torch.zeros(num_classes)
    for _, label in labeled_ds:
        labeled_class_counts[label] += 1
    logger.info(f"Labeled class distribution: {labeled_class_counts}")

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    weak_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

    strong_transform = v2.Compose([
        v2.RandAugment(num_ops=3, magnitude=5),
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

    labeled_ds = TransformedDataset(labeled_ds, transforms.ToTensor())
    unlabeled_ds = TransformedDatasetWithIndex(unlabeled_ds, transform=transforms.ToTensor())

    test_ds = TransformedDataset(test_ds, norm_transform)

    # ── DataLoader optimisé ────────────────────────────────────────────────────
    num_workers = min(2, os.cpu_count())  # ~2× CPU physiques
    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,  # transfert CPU→GPU DMA (plus rapide)
        prefetch_factor=4,  # pré-charge les batchs en avance
        persistent_workers=True,  # évite de respawn les workers à chaque epoch
    )
    # ───────────────────────────────────────────────────────────────────────────

    batch_size_l = 64
    mu = 7

    if optimized:
        labeled_loader = DataLoader(labeled_ds, batch_size=batch_size_l, shuffle=True, **dl_kwargs)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size_l * mu, shuffle=True, **dl_kwargs)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, **dl_kwargs)
    else:
        labeled_loader = DataLoader(labeled_ds, batch_size=batch_size_l, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size_l * mu, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_model(args.model, num_classes, args.widen_factor, args.depth)

    # ── Channels Last : layout NHWC optimal pour les Tensor Cores Ampere ───────
    model = model.to(device)
    if optimized:
        model = model.to(memory_format=torch.channels_last)
    # ───────────────────────────────────────────────────────────────────────────

    # --- EMA débrayable (use_ema=False -> évalue directement les poids en cours d'entraînement) ---
    if args.use_ema:
        eval_model = build_model(args.model, num_classes, args.widen_factor, args.depth).to(device)
        if optimized:
            eval_model = eval_model.to(memory_format=torch.channels_last)
        ema = EMA(model, args.ema_decay)
    else:
        eval_model = model
        ema = None

    # torch.compile() below wraps model in a module whose state_dict() keys are prefixed
    # (e.g. "_orig_mod.conv1.weight"), which would break EMA's key lookup -- keep a
    # reference to the uncompiled module (same underlying parameters) for EMA updates.
    base_model = model

    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    max_steps = args.max_steps
    total_steps = args.total_steps
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.03, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
    )
    scheduler = build_lr_scheduler(optimizer, total_steps, schedule=args.lr_schedule)

    # if optimized and presence of triton compiler, use torch.compile to optimize the model
    if optimized and torch.cuda.is_available() and "5060 Ti" in torch.cuda.get_device_name(0):
        try:
            import triton
            triton_available = True
        except ImportError:
            triton_available = False
        if triton_available:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile activé (mode=reduce-overhead)")

    # -- Hyper-parameters FixMatch ---------------------------------------------
    # tau=0.95 (confidence), mu=7 (unlabeled:labeled ratio), loss=Ls+Lu
    method_name = "fixmatch" + ("_ema" if args.use_ema else "") + (f"_wf{args.widen_factor}" if args.widen_factor != 2 else "")
    dataset_prefix = f"{args.dataset}-"
    name_of_experiment = f"{dataset_prefix}labeled-{num_labeled}-seed-{args.seed}"

    metrics = {
        "step": [],
        "train_loss": [],
        "test_f1": [],
        "test_acc": [],
        "time_elapsed": [],
        "pl_quality": [],
        "mask_ratio": [],
        "corrections": [],
        "new_errors": [],
        "error_reinforcement": [],
        "correct_reinforcement": [],
        "new_label": [],
        "new_correct": [],
        "bad_corrections": [],
        "losses": [],
        "topk_acc": []
    }

    last_pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    last_confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)

    true_labels = torch.tensor([label for _, label, _ in unlabeled_ds])
    pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)

    test_period = args.test_period

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results", name_of_experiment)
    os.makedirs(results_dir, exist_ok=True)
    start_time = time.time()
    mask_ratio = []
    losses = []

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    tau = args.tau
    verbose = args.verbose
    target_acc = args.target_acc

    for step in range(max_steps):
        model.train()

        try:
            x_l, y_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x_l, y_l = next(labeled_iter)

        try:
            x_u, y_u, idx = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_u, y_u, idx = next(unlabeled_iter)

        # Channels Last sur les inputs
        if optimized:
            x_l = weak_transform(x_l).to(device, non_blocking=True, memory_format=torch.channels_last)
            y_l = y_l.to(device, non_blocking=True)
            x_u_w = weak_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
            x_u_s = strong_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_l = weak_transform(x_l).to(device)
            y_l = y_l.to(device)
            x_u_w = weak_transform(x_u).to(device)
            x_u_s = strong_transform(x_u).to(device)

        # ── Pseudo-labels (no_grad + BF16) ────────────────────────────────────
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_u_w = model(x_u_w)
            probs_u_w = F.softmax(logits_u_w.float(), dim=1)  # softmax en FP32
            max_prob, pseudo = torch.max(probs_u_w, dim=1)
            # Keep only pseudo-labels with confidence >= tau
            mask = max_prob.ge(tau).float()
            mask_ratio.append(mask.mean().item())

        # ── Forward fusionné : labeled + strong unlabeled en un seul pass ─────
        optimizer.zero_grad(set_to_none=True)  # plus rapide que zero_grad()

        with autocast(device_type="cuda", dtype=torch.bfloat16) if optimized else nullcontext():
            # Un seul forward pour supervised + consistency
            batch_size_l_cur = x_l.shape[0]
            all_logits = model(torch.cat([x_l, x_u_s], dim=0))
            logits_l = all_logits[:batch_size_l_cur]
            logits_u_s = all_logits[batch_size_l_cur:]

            loss_supervised = F.cross_entropy(logits_l, y_l)
            loss_consistency = (
                mask * F.cross_entropy(logits_u_s, pseudo, reduction="none")
            ).mean()
            loss = loss_supervised + loss_consistency

            losses.append(loss.item())

        # ── Backward (pas de GradScaler nécessaire avec BF16) ─────────────────
        loss.backward()
        optimizer.step()
        scheduler.step()
        # ─────────────────────────────────────────────────────────────────────

        if ema is not None:
            ema.update(base_model)

        # update pseudo_labels passing throught the mask to avoid updating pseudo_labels for low confidence samples
        pseudo_labels[idx] = torch.where(mask.bool().cpu(), pseudo.cpu(), pseudo_labels[idx])
        confidences[idx] = torch.where(mask.bool().cpu(), max_prob.cpu(), confidences[idx])

        if (step + 1) % test_period == 0 or step == 0 or step == max_steps - 1:
            if ema is not None:
                ema.copy_to(eval_model)
            f1, acc, topk_accs = evaluate_f1_and_accuracy(eval_model, test_loader, device, args.topk)
            metrics["step"].append(step + 1)
            metrics["test_f1"].append(f1)
            metrics["test_acc"].append(acc)
            metrics["time_elapsed"].append(time.time() - start_time)
            metrics["mask_ratio"].append(np.mean(mask_ratio))
            metrics["train_loss"].append(np.mean(losses))
            metrics["topk_acc"].append(topk_accs)
            mask_ratio = []
            losses = []

            pl_metrics = compute_pseudo_label_metrics(
                pseudo_labels, confidences, last_pseudo_labels, last_confidences, true_labels
            )
            for key, value in pl_metrics.items():
                metrics[key].append(value)

            last_pseudo_labels = pseudo_labels.clone()
            last_confidences = confidences.clone()

            with open(f"{results_dir}/{method_name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            if verbose:
                # clear the in-place step line before logging the test-phase summary
                print()
            topk_str = " ".join(f"Top-{k}: {v:.4f}," for k, v in enumerate(topk_accs, start=1)) if args.topk > 1 else ""
            logger.info(
                f"Test F1: {f1:.4f}, Acc: {acc:.4f}, {topk_str} PL Quality: {pl_metrics['pl_quality']:.4f}, "
                f"Mask Ratio: {metrics['mask_ratio'][-1]:.4f}, Error Reinforcement: {pl_metrics['error_reinforcement']}, Correct Reinforcement: {pl_metrics['correct_reinforcement']}, "
                f"Corrections: {pl_metrics['corrections']}, Bad Corrections: {pl_metrics['bad_corrections']}, New Errors: {pl_metrics['new_errors']}, New Correct: {pl_metrics['new_correct']}, Loss: {metrics['train_loss'][-1]:.4f}, "
                f"Time: {metrics['time_elapsed'][-1]:.2f}s"
            )

            if target_acc is not None and acc >= target_acc:
                logger.info(f"Reached target_acc={target_acc:.4f} at step {step + 1} (acc={acc:.4f}) — stopping early.")
                break
        elif verbose:
            print(
                f"Step {step + 1}/{max_steps}, Loss: {loss.item():.4f}, "
                f"Sup: {loss_supervised.item():.4f}, Cons: {loss_consistency.item():.4f}, "
                f"Mask Ratio: {mask_ratio[-1]:.4f}",
                end="\r",
                flush=True,
            )


if __name__ == "__main__":
    run_fixmatch()
