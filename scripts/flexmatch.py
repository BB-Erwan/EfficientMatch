"""FlexMatch (Zhang et al., 2021) -- FixMatch with a per-class adaptive threshold.

Same two losses as FixMatch, but the confidence threshold is no longer the same for every class.
Curriculum Pseudo Labeling lowers it for classes the model has learned least, measured by how many
of their samples have ever been confidently pseudo-labeled:

  sigma(c)  = number of unlabeled samples confidently assigned to class c so far
  beta(c)   = sigma(c) / max over classes
  tau(c)    = tau * beta(c) / (2 - beta(c))

so a class nothing has been assigned to yet admits almost anything, and a well-learned class keeps
the full threshold. The intent is to stop easy classes from dominating early training.

On SVHN this mechanism does not converge: the threshold stays low enough that unreliable
pseudo-labels keep being admitted, and accuracy plateaus at 82-87% instead of reaching 90%. That is
a convergence failure rather than a speed one, and it reproduces what the FlexMatch authors report
on this dataset (see docs/main_experiment.md).

    python flexmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80
"""
import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.args import add_common_args, str2bool
from common.data import build_loaders, build_transforms, class_counts, load_datasets, split_labeled_unlabeled
from common.recording import evaluate_and_log, new_metrics, results_path, track_pseudo_labels
from common.setup import build_model_and_optimizer, enable_runtime_optimizations, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--tau", type=float, default=0.95,
                    help="Base confidence threshold, scaled down per class by the curriculum.")
parser.add_argument("--thresh_warmup", type=str2bool, default=True,
                    help="Count still-unassigned samples in the per-class normalisation during warmup, as "
                         "the reference implementation does. Turning it off makes the threshold rise much "
                         "earlier and changes convergence markedly.")
args = parser.parse_args()

MU = 7   # unlabeled:labeled ratio, as established by the original paper


def run_flexmatch():
    enable_runtime_optimizations()
    device = select_device()

    num_classes, mean, std, train_ds, test_ds = load_datasets(args.dataset)
    logger.info(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    torch.manual_seed(args.seed)
    labeled_ds, unlabeled_ds = split_labeled_unlabeled(train_ds, num_classes, args.num_labeled)
    logger.info(f"Labeled class distribution: {class_counts(labeled_ds, num_classes)}")

    norm_transform, weak_transform, strong_transform = build_transforms(mean, std)
    batch_size_l = 64
    labeled_loader, unlabeled_loader, test_loader, unlabeled_ds = build_loaders(
        labeled_ds, unlabeled_ds, test_ds, norm_transform, batch_size_l, MU, args.optimized)

    model, base_model, eval_model, ema, optimizer, scheduler = build_model_and_optimizer(
        args, num_classes, device)

    method_name = ("flexmatch"
                   + ("_ema" if args.use_ema else "")
                   + (f"_wf{args.widen_factor}" if args.widen_factor != 2 else "")
                   + (f"_{args.tag}" if args.tag else ""))
    path = results_path(args, method_name)

    # Curriculum state: the class every unlabeled sample has been confidently assigned to so far
    # (-1 when none), and the per-class learning estimate derived from it.
    selected_label = torch.full((len(unlabeled_ds),), -1, dtype=torch.long, device=device)
    classwise_acc = torch.zeros(num_classes, dtype=torch.float32, device=device)

    metrics = new_metrics()
    pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)
    pseudo_state = (pseudo_labels, confidences, pseudo_labels.clone(), confidences.clone(),
                    torch.tensor([label for _, label, _ in unlabeled_ds]))

    losses, mask_ratios = [], []
    start_time = time.time()
    use_autocast = args.optimized and device.type == "cuda"
    labeled_iter, unlabeled_iter = iter(labeled_loader), iter(unlabeled_loader)

    for step in range(args.max_steps):
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

        if args.optimized:
            x_l = weak_transform(x_l).to(device, non_blocking=True, memory_format=torch.channels_last)
            y_l = y_l.to(device, non_blocking=True)
            x_u_w = weak_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
            x_u_s = strong_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_l, y_l = weak_transform(x_l).to(device), y_l.to(device)
            x_u_w, x_u_s = weak_transform(x_u).to(device), strong_transform(x_u).to(device)
        idx_device = idx.to(device, non_blocking=args.optimized)

        # --- Pseudo-labels and the per-class threshold. ---
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else nullcontext():
                logits_u_w = model(x_u_w)
            probs_u_w = F.softmax(logits_u_w.float(), dim=1)
            max_prob, pseudo = torch.max(probs_u_w, dim=-1)

            acc_per_sample = classwise_acc[pseudo]
            mask = max_prob.ge(args.tau * (acc_per_sample / (2.0 - acc_per_sample))).float()

            select = max_prob.ge(args.tau)
            if select.any():
                selected_label[idx_device[select]] = pseudo[select]

            # Bin 0 counts the still-unselected (-1) samples, bins 1.. the per-class counts.
            # thresh_warmup relies on bin 0 dominating early training -- it is huge relative to any
            # real class -- to keep classwise_acc near 0, and the mask permissive, until enough
            # samples have been confidently pseudo-labeled. It must not be filtered out beforehand.
            counts = torch.bincount(selected_label + 1, minlength=num_classes + 1)
            if counts.max().item() < selected_label.shape[0]:
                if args.thresh_warmup:
                    denom = max(counts.max().item(), 1)
                else:
                    without_unselected = counts.clone()
                    without_unselected[0] = 0
                    denom = max(without_unselected.max().item(), 1)
                classwise_acc = counts[1:].float() / denom

            mask_ratios.append(mask.mean().item())

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else nullcontext():
            n_l = x_l.shape[0]
            all_logits = model(torch.cat([x_l, x_u_s], dim=0))
            loss_supervised = F.cross_entropy(all_logits[:n_l], y_l)
            loss_consistency = (mask * F.cross_entropy(all_logits[n_l:], pseudo, reduction="none")).mean()
            loss = loss_supervised + loss_consistency
            losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(base_model)

        track_pseudo_labels(pseudo_labels, confidences, idx, mask, pseudo, max_prob)

        if (step + 1) % args.test_period == 0 or step == 0 or step == args.max_steps - 1:
            if ema is not None:
                ema.copy_to(eval_model)
            stop = evaluate_and_log(args, step, metrics, path, eval_model, test_loader, device,
                                    start_time, losses, mask_ratios, pseudo_state)
            losses, mask_ratios = [], []
            if stop:
                break
        elif args.verbose:
            print(f"Step {step + 1}/{args.max_steps}, Loss: {loss.item():.4f}, "
                  f"Sup: {loss_supervised.item():.4f}, Cons: {loss_consistency.item():.4f}, "
                  f"Mask Ratio: {mask_ratios[-1]:.4f}", end="\r", flush=True)


if __name__ == "__main__":
    run_flexmatch()
