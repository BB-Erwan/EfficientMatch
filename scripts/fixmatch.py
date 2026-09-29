"""FixMatch (Sohn et al., 2020) -- the consistency baseline.

Two losses. A supervised cross-entropy on the labeled batch, and a consistency loss that asks the
prediction on a strongly augmented unlabeled view to match the pseudo-label taken from the weakly
augmented one -- but only for samples whose pseudo-label is confident enough:

  L_s   cross-entropy on the labeled batch
  L_u   cross-entropy between the pseudo-label of U_w and the prediction on U_s, masked by
        confidence >= tau

  L = L_s + L_u

Confidence thresholding is what keeps unreliable pseudo-labels out of the loss, and also what makes
FixMatch slow early in training: at the start almost nothing passes the threshold, so most of each
unlabeled batch contributes nothing. EfficientMatch keeps this loss unchanged and reuses its mask
for a third term; see efficientmatch.py.

    python fixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80
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

from common.args import add_common_args
from common.data import build_loaders, build_transforms, class_counts, load_datasets, split_labeled_unlabeled
from common.recording import evaluate_and_log, new_metrics, results_path, track_pseudo_labels
from common.setup import build_model_and_optimizer, enable_runtime_optimizations, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--mu", type=int, default=7,
                    help="Unlabeled:labeled batch size ratio. 7 is the value established by the original "
                         "paper and kept here; Appendix A.6 also reports FixMatch at mu = 3.")
parser.add_argument("--tau", type=float, default=0.95, help="Confidence threshold on the pseudo-label.")
args = parser.parse_args()


def run_fixmatch():
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
        labeled_ds, unlabeled_ds, test_ds, norm_transform, batch_size_l, args.mu, args.optimized)

    model, base_model, eval_model, ema, optimizer, scheduler = build_model_and_optimizer(
        args, num_classes, device)

    method_name = ("fixmatch"
                   + (f"_mu{args.mu}" if args.mu != 7 else "")
                   + ("_ema" if args.use_ema else "")
                   + (f"_wf{args.widen_factor}" if args.widen_factor != 2 else "")
                   + (f"_{args.tag}" if args.tag else ""))
    path = results_path(args, method_name)

    metrics = new_metrics()
    pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)
    pseudo_state = (pseudo_labels, confidences, pseudo_labels.clone(), confidences.clone(),
                    torch.tensor([label for _, label, _ in unlabeled_ds]))

    losses, mask_ratios = [], []
    start_time = time.time()
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

        # --- Pseudo-labels, without gradient: one forward pass on the weak view. ---
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_u_w = model(x_u_w)
            probs_u_w = F.softmax(logits_u_w.float(), dim=1)
            max_prob, pseudo = torch.max(probs_u_w, dim=1)
            mask = max_prob.ge(args.tau).float()
            mask_ratios.append(mask.mean().item())

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16) if args.optimized else nullcontext():
            # One fused forward pass for both losses: the labeled batch and the strong view.
            n_l = x_l.shape[0]
            all_logits = model(torch.cat([x_l, x_u_s], dim=0))
            loss_supervised = F.cross_entropy(all_logits[:n_l], y_l)
            loss_consistency = (mask * F.cross_entropy(all_logits[n_l:], pseudo, reduction="none")).mean()
            loss = loss_supervised + loss_consistency
            losses.append(loss.item())

        loss.backward()   # bfloat16 needs no GradScaler
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
    run_fixmatch()
