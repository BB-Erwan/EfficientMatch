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
from torch.utils.data import DataLoader
from torchvision.transforms import v2

sys.path.append("..")  # add parent directory to path for imports

from datasets_utils import TransformedDataset
from models import build_model
from utils import evaluate_f1_and_accuracy, build_lr_scheduler
from ema import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Fully-supervised baseline (full labeled training set, no pseudo-labeling) to benchmark a backbone's raw classification ceiling.")
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "svhn"])
parser.add_argument("--widen_factor", type=int, default=8, help="WideResNet-28-{widen_factor}. Papers use 2 for CIFAR-10 and 8 for CIFAR-100.")
parser.add_argument("--model", type=str, default="densenet", choices=["wideresnet", "densenet"], help="Backbone architecture.")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="SGD weight decay. Papers use 5e-4 for CIFAR-10 and 1e-3 for CIFAR-100.")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.1, help="Base SGD learning rate for the supervised baseline.")
parser.add_argument("--optimized", type=str2bool, default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test_period", type=int, default=500)
parser.add_argument("--max_steps", type=int, default=2**16, help="Number of steps actually run; the run is truncated here.")
parser.add_argument("--total_steps", type=int, default=2**16, help="Nominal horizon the cosine LR schedule decays over, independent of max_steps.")
parser.add_argument("--lr_schedule", type=str, default="fixmatch_cosine", choices=["fixmatch_cosine", "cosine_annealing"], help="LR schedule: rescaled FixMatch cosine (default) or torch's classic CosineAnnealingLR.")
parser.add_argument("--verbose", type=str2bool, default=False)
parser.add_argument("--target_acc", type=float, default=None, help="Stop the run early once test_acc reaches this value.")
parser.add_argument("--use_ema", type=str2bool, default=True, help="Evaluate an EMA of the weights instead of the raw training weights.")
parser.add_argument("--ema_decay", type=float, default=0.999)
args = parser.parse_args()


def run_supervised():
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

    optimized = args.optimized
    torch.manual_seed(args.seed)

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

    train_ds = TransformedDataset(train_ds, transforms.ToTensor())
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

    batch_size = args.batch_size

    if optimized:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, **dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_model(args.model, num_classes, args.widen_factor)

    # ── Channels Last : layout NHWC optimal pour les Tensor Cores Ampere ───────
    model = model.to(device)
    if optimized:
        model = model.to(memory_format=torch.channels_last)
    # ───────────────────────────────────────────────────────────────────────────

    # --- EMA débrayable (use_ema=False -> évalue directement les poids en cours d'entraînement) ---
    if args.use_ema:
        eval_model = build_model(args.model, num_classes, args.widen_factor).to(device)
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
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
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

    method_name = f"supervised_{args.model}" + ("_ema" if args.use_ema else "")
    dataset_prefix = f"{args.dataset}-"
    name_of_experiment = f"{dataset_prefix}supervised-seed-{args.seed}"

    metrics = {
        "train_loss": [],
        "test_f1": [],
        "test_acc": [],
        "time_elapsed": [],
    }

    test_period = args.test_period

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results", name_of_experiment)
    os.makedirs(results_dir, exist_ok=True)
    start_time = time.time()
    losses = []

    train_iter = iter(train_loader)

    verbose = args.verbose
    target_acc = args.target_acc

    for step in range(max_steps):
        model.train()

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        if optimized:
            x = weak_transform(x).to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
        else:
            x = weak_transform(x).to(device)
            y = y.to(device)

        optimizer.zero_grad(set_to_none=True)  # plus rapide que zero_grad()

        with autocast(device_type="cuda", dtype=torch.bfloat16) if optimized else nullcontext():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(base_model)

        if (step + 1) % test_period == 0 or step == 0 or step == max_steps - 1:
            if ema is not None:
                ema.copy_to(eval_model)
            f1, acc = evaluate_f1_and_accuracy(eval_model, test_loader, device)
            metrics["test_f1"].append(f1)
            metrics["test_acc"].append(acc)
            metrics["time_elapsed"].append(time.time() - start_time)
            metrics["train_loss"].append(np.mean(losses))
            losses = []

            with open(f"{results_dir}/{method_name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            if verbose:
                # clear the in-place step line before logging the test-phase summary
                print()
            logger.info(
                f"Test F1: {f1:.4f}, Acc: {acc:.4f}, Loss: {metrics['train_loss'][-1]:.4f}, "
                f"Time: {metrics['time_elapsed'][-1]:.2f}s"
            )

            if target_acc is not None and acc >= target_acc:
                logger.info(f"Reached target_acc={target_acc:.4f} at step {step + 1} (acc={acc:.4f}) — stopping early.")
                break
        elif verbose:
            print(
                f"Step {step + 1}/{max_steps}, Loss: {loss.item():.4f}",
                end="\r",
                flush=True,
            )


if __name__ == "__main__":
    run_supervised()
