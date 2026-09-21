"""Fast FixMatch (Chen, Dun & Kyrillidis, arXiv:2309.03469): FixMatch + Curriculum Batch Size
(CBS, "B-EXP" schedule). Copy of fixmatch.py with exactly two differences:

  1. The unlabeled batch size is NOT fixed at mu*batch_size_l -- it varies every step following
     B(t) = u * (1 - (1 - t/T) / ((1 - alpha) + alpha * (1 - t/T))), alpha=0.7, where u is the
     nominal max (mu*batch_size_l), t is the current step and T is --cbs_total_steps (the
     curriculum horizon, DELIBERATELY DECOUPLED from --total_steps: the latter is this repo's
     oversized nominal LR-schedule horizon (2**20), and coupling the CBS ramp to it would leave
     the batch size stuck near its floor for the entire realistic length of a run here -- exactly
     the bug found and fixed in mixmatch_2.py's lambda_u ramp). The full mu*batch_size_l batch is
     still drawn from the DataLoader every step and simply truncated to the CBS-prescribed size,
     rather than building a variable-batch-size DataLoader.
  2. The unsupervised loss weight is lambda = (current unlabeled batch size) / batch_size_l (the
     paper's rule), instead of fixmatch's implicit weight of 1.0.

This script also measures cumulative FLOPs (labeled+unlabeled forward/backward samples processed,
scaled by two one-time FlopCounterMode calibration measurements -- FLOPs scale exactly linearly
with batch size for a fixed CNN forward/backward, so two reference measurements are enough to
price any batch size, no need to remeasure per shape) alongside wall-clock time_elapsed, precisely
to let a short run be compared against a fixmatch run of the same step budget on both axes: FLOPs
(should favor Fast FixMatch, confirming the paper) and wall-clock time (may not, if the varying
tensor shapes defeat this pipeline's cudnn.benchmark/torch.compile assumptions -- see
ABLATION_FAST_FIXMATCH.md).
"""
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
from torch.utils.flop_counter import FlopCounterMode
from torchvision.transforms import v2

sys.path.append("..")  # add parent directory to path for imports

from datasets_utils import TransformedDataset, TransformedDatasetWithIndex
from models import build_model
from utils import evaluate_f1_and_accuracy, build_lr_scheduler
from ema import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_pseudo_label_metrics(pseudo_labels, confidences, last_pseudo_labels, last_confidences, true_labels):
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


def cbs_unlabeled_batch_size(step, total_steps, max_batch, alpha, min_batch):
    """B-EXP Curriculum Batch Size schedule (see module docstring)."""
    frac_left = 1.0 - step / total_steps
    denom = (1 - alpha) + alpha * frac_left
    raw = max_batch * (1.0 - frac_left / denom)
    return int(max(min_batch, min(max_batch, round(raw))))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
parser.add_argument("--widen_factor", type=int, default=2, help="WideResNet-28-{widen_factor}. Papers use 2 for CIFAR-10 and 8 for CIFAR-100.")
parser.add_argument("--model", type=str, default="wideresnet", choices=["wideresnet", "resnet18"], help="Backbone architecture.")
parser.add_argument("--depth", type=int, default=28, help="WideResNet depth (e.g. 28 for WRN-28-x, 40 for WRN-40-x). Ignored for resnet18. Must satisfy (depth-4) mod 6 == 0.")
parser.add_argument("--topk", type=int, default=1, help="Also report top-k accuracy for every k from 1 to this value.")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="SGD weight decay. Papers use 5e-4 for CIFAR-10 and 1e-3 for CIFAR-100.")
parser.add_argument("--num_labeled", type=int, default=250)
parser.add_argument("--optimized", type=str2bool, default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test_period", type=int, default=500)
parser.add_argument("--tau", type=float, default=0.95)
parser.add_argument("--max_steps", type=int, default=2**20, help="Number of steps actually run; the run is truncated here.")
parser.add_argument("--total_steps", type=int, default=2**20, help="Nominal horizon the cosine LR schedule decays over, independent of max_steps.")
parser.add_argument("--lr_schedule", type=str, default="fixmatch_cosine", choices=["fixmatch_cosine", "cosine_annealing"])
parser.add_argument("--verbose", type=str2bool, default=False)
parser.add_argument("--target_acc", type=float, default=None, help="Stop the run early once test_acc reaches this value.")
parser.add_argument("--use_ema", type=str2bool, default=True, help="Evaluate an EMA of the weights instead of the raw training weights.")
parser.add_argument("--mu", type=int, default=7, help="Unlabeled:labeled batch size ratio -> defines the MAX (not fixed) unlabeled batch size for CBS.")
parser.add_argument("--ema_decay", type=float, default=0.999)
parser.add_argument("--alpha", type=float, default=0.7, help="CBS B-EXP shape parameter (0.7 in the Fast FixMatch paper).")
parser.add_argument("--cbs_min_batch", type=int, default=8, help="Floor on the unlabeled batch size.")
parser.add_argument("--cbs_total_steps", type=int, default=60000, help="Curriculum horizon T for the CBS ramp -- decoupled from --total_steps, see module docstring.")
args = parser.parse_args()


def run_fast_fixmatch():
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
    logger.info(f"Mean: {mean}, Std: {std}")

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

    labeled_class_counts = torch.zeros(num_classes)
    for _, label in labeled_ds:
        labeled_class_counts[label] += 1
    logger.info(f"Labeled class distribution: {labeled_class_counts}")

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
    mu = args.mu
    max_u_batch = batch_size_l * mu

    if optimized:
        labeled_loader = DataLoader(labeled_ds, batch_size=batch_size_l, shuffle=True, **dl_kwargs)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=max_u_batch, shuffle=True, **dl_kwargs)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, **dl_kwargs)
    else:
        labeled_loader = DataLoader(labeled_ds, batch_size=batch_size_l, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=max_u_batch, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_model(args.model, num_classes, args.widen_factor, args.depth)
    model = model.to(device)
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

    max_steps = args.max_steps
    total_steps = args.total_steps
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = build_lr_scheduler(optimizer, total_steps, schedule=args.lr_schedule)

    if optimized and torch.cuda.is_available() and "5060 Ti" in torch.cuda.get_device_name(0):
        try:
            import triton
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile activé (mode=reduce-overhead)")
        except ImportError:
            pass

    # --- FLOPs calibration (once, before training): FLOPs scale exactly linearly with batch size
    # for a fixed CNN forward/backward, so two reference measurements price any CBS batch size
    # without remeasuring per shape (see module docstring). ---
    calib_bs = 64
    dummy_x = torch.randn(calib_bs, 3, 32, 32, device=device)
    dummy_y = torch.randint(0, num_classes, (calib_bs,), device=device)
    base_model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            base_model(dummy_x)
    flops_per_sample_fwd = fc.get_total_flops() / calib_bs
    with FlopCounterMode(display=False) as fc:
        logits = base_model(dummy_x)
        F.cross_entropy(logits, dummy_y).backward()
    flops_per_sample_fwd_bwd = fc.get_total_flops() / calib_bs
    base_model.zero_grad(set_to_none=True)
    logger.info(f"FLOPs calibration: {flops_per_sample_fwd:.3e} FLOPs/sample (fwd only), "
                f"{flops_per_sample_fwd_bwd:.3e} FLOPs/sample (fwd+bwd)")

    def step_gflops(u_bs):
        total = u_bs * flops_per_sample_fwd + (batch_size_l + u_bs) * flops_per_sample_fwd_bwd
        return total / 1e9

    method_name = "fast_fixmatch" + ("_ema" if args.use_ema else "") + (f"_wf{args.widen_factor}" if args.widen_factor != 2 else "")
    dataset_prefix = f"{args.dataset}-"
    name_of_experiment = f"{dataset_prefix}labeled-{num_labeled}-seed-{args.seed}"

    metrics = {
        "step": [], "train_loss": [], "test_f1": [], "test_acc": [], "time_elapsed": [],
        "pl_quality": [], "mask_ratio": [], "corrections": [], "new_errors": [],
        "error_reinforcement": [], "correct_reinforcement": [], "new_label": [], "new_correct": [],
        "bad_corrections": [], "topk_acc": [], "cumulative_gflops": [], "u_batch_size": [],
    }

    last_pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    last_confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)
    true_labels = torch.tensor([label for _, label, _ in unlabeled_ds])
    pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)

    test_period = args.test_period
    tau = args.tau
    verbose = args.verbose
    target_acc = args.target_acc

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results", name_of_experiment)
    os.makedirs(results_dir, exist_ok=True)
    start_time = time.time()
    mask_ratio = []
    losses = []
    cumulative_gflops = 0.0

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

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

        # --- Curriculum Batch Size: truncate the max-size unlabeled batch to the size the B-EXP
        # schedule prescribes for this step (see module docstring). ---
        u_bs = cbs_unlabeled_batch_size(step, args.cbs_total_steps, max_u_batch, args.alpha, args.cbs_min_batch)
        u_bs = min(u_bs, x_u.shape[0])
        x_u = x_u[:u_bs]
        idx = idx[:u_bs]
        cumulative_gflops += step_gflops(u_bs)

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

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16) if optimized else nullcontext():
                logits_u_w = model(x_u_w)
            probs_u_w = F.softmax(logits_u_w.float(), dim=1)
            max_prob, pseudo = torch.max(probs_u_w, dim=1)
            mask = max_prob.ge(tau).float()
            mask_ratio.append(mask.mean().item())

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16) if optimized else nullcontext():
            n_l = x_l.shape[0]
            all_logits = model(torch.cat([x_l, x_u_s], dim=0))
            logits_l = all_logits[:n_l]
            logits_u_s = all_logits[n_l:]

            loss_supervised = F.cross_entropy(logits_l, y_l)
            # Fast FixMatch's rule: unsupervised weight = current unlabeled batch size / labeled
            # batch size (u_bs/batch_size_l), instead of fixmatch's implicit weight of 1.0.
            lambda_u = u_bs / batch_size_l
            loss_consistency = lambda_u * (mask * F.cross_entropy(logits_u_s, pseudo, reduction="none")).mean()
            loss = loss_supervised + loss_consistency
            losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(base_model)

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
            metrics["mask_ratio"].append(float(np.mean(mask_ratio)) if mask_ratio else 0.0)
            metrics["train_loss"].append(float(np.mean(losses)) if losses else 0.0)
            metrics["topk_acc"].append(topk_accs)
            metrics["cumulative_gflops"].append(cumulative_gflops)
            metrics["u_batch_size"].append(u_bs)
            mask_ratio = []
            losses = []

            pl_metrics = compute_pseudo_label_metrics(pseudo_labels, confidences, last_pseudo_labels, last_confidences, true_labels)
            for key, value in pl_metrics.items():
                metrics[key].append(value)
            last_pseudo_labels = pseudo_labels.clone()
            last_confidences = confidences.clone()

            with open(f"{results_dir}/{method_name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            if verbose:
                print()
            topk_str = " ".join(f"Top-{k}: {v:.4f}," for k, v in enumerate(topk_accs, start=1)) if args.topk > 1 else ""
            logger.info(
                f"Test F1: {f1:.4f}, Acc: {acc:.4f}, {topk_str} PL Quality: {pl_metrics['pl_quality']:.4f}, "
                f"Mask Ratio: {metrics['mask_ratio'][-1]:.4f}, u_batch_size: {u_bs}, "
                f"Cumulative GFLOPs: {cumulative_gflops:.1f}, Loss: {metrics['train_loss'][-1]:.4f}, "
                f"Time: {metrics['time_elapsed'][-1]:.2f}s"
            )

            if target_acc is not None and acc >= target_acc:
                logger.info(f"Reached target_acc={target_acc:.4f} at step {step + 1} (acc={acc:.4f}) — stopping early.")
                break
        elif verbose:
            print(f"Step {step + 1}/{max_steps}, u_batch_size={u_bs}, Loss: {loss.item():.4f}", end="\r", flush=True)


if __name__ == "__main__":
    run_fast_fixmatch()
