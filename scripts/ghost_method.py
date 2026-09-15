"""'Ghost' method: exact same harness as the real training scripts (dataset split, weak/strong
augmentations, dataloaders, model + EMA construction, torch.compile activation, evaluation call
and metrics/JSON logging) but the semi-supervised training step itself does nothing -- no model
forward, no backward, no optimizer step. The only GPU compute that happens is the periodic
evaluation call, run in situ (interleaved with real dataloader/augmentation traffic, real EMA
weight copies) exactly like a real run.

Purpose: sanity-check the isolated eval-time measurement from measure_eval_time.py against the
same call inside the real per-run harness, with a low --test_period so evaluations dominate the
timeline and n_evals * eval_time can be read almost directly off time_elapsed.

Usage:
    python ghost_method.py --dataset cifar10 --num_labeled 250 --seed 2312 --test_period 20 --max_steps 200
    python ghost_method.py --dataset cifar100 --widen_factor 8 --test_period 20 --max_steps 200
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

sys.path.append("..")

from datasets_utils import TransformedDataset, TransformedDatasetWithIndex
from models import build_model
from utils import evaluate_f1_and_accuracy
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


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
parser.add_argument("--widen_factor", type=int, default=2)
parser.add_argument("--model", type=str, default="wideresnet", choices=["wideresnet", "resnet18"])
parser.add_argument("--depth", type=int, default=28)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--num_labeled", type=int, default=250)
parser.add_argument("--optimized", type=str2bool, default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test_period", type=int, default=20, help="Kept low by default so evaluations dominate the timeline.")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--use_ema", type=str2bool, default=True)
parser.add_argument("--ema_decay", type=float, default=0.999)
args = parser.parse_args()


def run_ghost():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

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

    num_labeled = args.num_labeled
    optimized = args.optimized
    torch.manual_seed(args.seed)
    train_ds = Subset(train_ds, torch.randperm(len(train_ds)))

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

    norm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    weak_transform = v2.Compose([
        v2.ToImage(), v2.RandomHorizontalFlip(), v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std),
    ])
    strong_transform = v2.Compose([
        v2.RandAugment(num_ops=3, magnitude=5),
        v2.ToImage(), v2.RandomHorizontalFlip(), v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std),
    ])

    labeled_ds = TransformedDataset(labeled_ds, transforms.ToTensor())
    unlabeled_ds = TransformedDatasetWithIndex(unlabeled_ds, transform=transforms.ToTensor())
    test_ds = TransformedDataset(test_ds, norm_transform)

    num_workers = min(2, os.cpu_count())
    dl_kwargs = dict(num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)

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

    model = build_model(args.model, num_classes, args.widen_factor, args.depth).to(device)
    if optimized:
        model = model.to(memory_format=torch.channels_last)

    if args.use_ema:
        eval_model = build_model(args.model, num_classes, args.widen_factor, args.depth).to(device)
        if optimized:
            eval_model = eval_model.to(memory_format=torch.channels_last)
        ema = EMA(model, args.ema_decay)
    else:
        eval_model = model
        ema = None
    base_model = model

    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # torch.compile is activated for parity with the real scripts even though the ghost loop
    # never calls model() -- it costs nothing here since the wrapped model is simply unused.
    if optimized and torch.cuda.is_available() and "5060 Ti" in torch.cuda.get_device_name(0):
        try:
            import triton  # noqa: F401
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile activé (mode=reduce-overhead)")
        except ImportError:
            pass

    method_name = "ghost" + ("_ema" if args.use_ema else "")
    dataset_prefix = f"{args.dataset}-"
    name_of_experiment = f"{dataset_prefix}labeled-{num_labeled}-seed-{args.seed}"

    metrics = {"step": [], "test_f1": [], "test_acc": [], "time_elapsed": [], "topk_acc": [], "eval_time_s": []}

    test_period = args.test_period
    max_steps = args.max_steps

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results", name_of_experiment)
    os.makedirs(results_dir, exist_ok=True)
    start_time = time.time()

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for step in range(max_steps):
        model.train()

        # Keep the exact same dataloader/augmentation/host->device traffic as a real run --
        # only the model forward/backward/optimizer step is skipped ("does nothing").
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

        # -- no model(), no loss, no backward, no optimizer.step() --

        if ema is not None:
            ema.update(base_model)

        if (step + 1) % test_period == 0 or step == 0 or step == max_steps - 1:
            torch.cuda.synchronize() if device.type == "cuda" else None
            eval_t0 = time.perf_counter()
            if ema is not None:
                ema.copy_to(eval_model)
            f1, acc, topk_accs = evaluate_f1_and_accuracy(eval_model, test_loader, device, args.topk)
            torch.cuda.synchronize() if device.type == "cuda" else None
            eval_time_s = time.perf_counter() - eval_t0

            metrics["step"].append(step + 1)
            metrics["test_f1"].append(f1)
            metrics["test_acc"].append(acc)
            metrics["time_elapsed"].append(time.time() - start_time)
            metrics["topk_acc"].append(topk_accs)
            metrics["eval_time_s"].append(eval_time_s)

            with open(f"{results_dir}/{method_name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"[ghost] step={step + 1} acc={acc:.4f} eval_time={eval_time_s*1000:.1f}ms time_elapsed={metrics['time_elapsed'][-1]:.2f}s")

    mean_eval_time = sum(metrics["eval_time_s"][1:]) / max(len(metrics["eval_time_s"]) - 1, 1)  # skip first (cuDNN autotune warmup)
    logger.info(f"[ghost] mean eval_time over {len(metrics['eval_time_s']) - 1} evals (excl. first): {mean_eval_time*1000:.1f}ms")


if __name__ == "__main__":
    run_ghost()
