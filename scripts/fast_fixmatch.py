"""Fast FixMatch (Chen et al., 2024) -- FixMatch with a Curriculum Batch Size.

The losses are FixMatch's, unchanged. What changes is how many unlabeled samples each iteration
uses: instead of a fixed mu*B, the batch grows over training along the B-EXP schedule

  B(t) = u * (1 - (1 - t/T) / ((1 - alpha) + alpha * (1 - t/T)))

where u is the nominal maximum (mu * batch_size_l), t the current iteration and T the curriculum
horizon --cbs_total_steps. The idea is that early iterations, whose pseudo-labels are unreliable
anyway, do not deserve a full batch.

The horizon is deliberately decoupled from --total_steps: the learning-rate schedule keeps decaying
over 2^20 iterations as for every other method, while the batch-size curriculum sweeps its whole
range over a much shorter span, which is what the diagnostic of Appendix A.4 measures.

This method is measured but not entered in the main comparison. Varying the batch size every
iteration means varying tensor shapes, and torch.compile in reduce-overhead mode records a new CUDA
graph for each distinct shape: the 62.6% saving in FLOPs turns into a 2.13x *loss* in wall-clock
time on this pipeline. See docs/fast_fixmatch.md.

--full adds the two other components of Fast FixMatch as defined by its authors (Algorithm 1 of
Chen et al.): strong augmentation of the labeled batch (RandAugment here, AutoAugment in the paper)
and FlexMatch's Curriculum Pseudo Labeling with the convex mapping, and scales the unlabeled loss
weight with the current unlabeled batch. Chen et al. write lambda = u_bs / labeled batch
(--lambda_ref labeled); the default --lambda_ref max uses u_bs / max batch (lambda = 1 at the full
batch, as in FixMatch), which is 7x smaller and was our first reading. Use it with --cbs_total_steps 1048576, the horizon of the paper (the whole run).

Because the point of this method is the compute it saves, it also records the FLOPs it actually
spends, calibrated once on the real model rather than assumed.

    python scripts/benchmark_fast_fixmatch.py      # the FixMatch-vs-CBS diagnostic of Table 9
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
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.args import add_common_args, str2bool
from common.data import build_loaders, build_transforms, class_counts, image_size, load_datasets, split_labeled_unlabeled
from common.recording import evaluate_and_log, new_metrics, results_path, track_pseudo_labels
from common.setup import build_model_and_optimizer, enable_runtime_optimizations, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--mu", type=int, default=7, help="Sets the nominal maximum unlabeled batch, mu * 64.")
parser.add_argument("--tau", type=float, default=0.95, help="Confidence threshold, as in FixMatch.")
parser.add_argument("--alpha", type=float, default=0.7, help="Curvature of the B-EXP schedule.")
parser.add_argument("--cbs_min_batch", type=int, default=8, help="Floor on the unlabeled batch size.")
parser.add_argument("--cbs_total_steps", type=int, default=60000,
                    help="Curriculum horizon T, decoupled from --total_steps (see module docstring).")
parser.add_argument("--full", type=str2bool, default=False,
                    help="Full Fast FixMatch: CBS + strong labeled augmentation + FlexMatch CPL + "
                         "lambda scaled with the unlabeled batch. Results get a '_full' suffix.")
parser.add_argument("--lambda_ref", choices=["max", "labeled"], default="max",
                    help="With --full, lambda = u_bs / reference. 'labeled' divides by the labeled batch "
                         "(64), as written in Chen et al. (lambda = 96/64 = 1.5 for an unlabeled batch of 96; "
                         "suffix '_lamlab'). 'max' divides by mu*64, so lambda = 1 at the full batch; it was "
                         "the first version and is kept so existing results stay reproducible.")
parser.add_argument("--compile", type=str2bool, default=True,
                    help="torch.compile the model. False runs in eager mode, where varying batch shapes "
                         "cost no recompilation; results then get an '_eager' suffix.")
args = parser.parse_args()


def cbs_unlabeled_batch_size(step, total_steps, max_batch, alpha, min_batch):
    """B-EXP Curriculum Batch Size schedule, clamped to [min_batch, max_batch]."""
    frac_left = 1.0 - step / total_steps
    raw = max_batch * (1.0 - frac_left / ((1 - alpha) + alpha * frac_left))
    return int(max(min_batch, min(max_batch, round(raw))))


def run_fast_fixmatch():
    enable_runtime_optimizations()
    device = select_device()

    num_classes, mean, std, train_ds, test_ds = load_datasets(args.dataset)
    logger.info(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    torch.manual_seed(args.seed)
    labeled_ds, unlabeled_ds = split_labeled_unlabeled(train_ds, num_classes, args.num_labeled)
    logger.info(f"Labeled class distribution: {class_counts(labeled_ds, num_classes)}")

    norm_transform, weak_transform, strong_transform = build_transforms(mean, std, image_size(args.dataset))
    batch_size_l = 64
    max_u_batch = batch_size_l * args.mu
    labeled_loader, unlabeled_loader, test_loader, unlabeled_ds = build_loaders(
        labeled_ds, unlabeled_ds, test_ds, norm_transform, batch_size_l, args.mu, args.optimized)

    model, base_model, eval_model, ema, optimizer, scheduler = build_model_and_optimizer(
        args, num_classes, device, compile_model=args.compile)

    # Calibrate the per-sample cost once on the real model, so the FLOPs this run reports are
    # measured rather than assumed. Forward-only for the pseudo-label pass, forward+backward for the
    # rest; both scale linearly in batch size, which is what makes a variable batch size meaningful.
    calib_bs = 8
    dummy_x = torch.randn(calib_bs, 3, 32, 32, device=device)
    dummy_y = torch.randint(0, num_classes, (calib_bs,), device=device)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            base_model(dummy_x)
    flops_per_sample_fwd = fc.get_total_flops() / calib_bs
    with FlopCounterMode(display=False) as fc:
        F.cross_entropy(base_model(dummy_x), dummy_y).backward()
    flops_per_sample_fwd_bwd = fc.get_total_flops() / calib_bs
    base_model.zero_grad(set_to_none=True)
    logger.info(f"FLOPs calibration: {flops_per_sample_fwd:.3e} FLOPs/sample (fwd only), "
                f"{flops_per_sample_fwd_bwd:.3e} FLOPs/sample (fwd+bwd)")

    def step_gflops(u_bs):
        return (u_bs * flops_per_sample_fwd + (batch_size_l + u_bs) * flops_per_sample_fwd_bwd) / 1e9

    method_name = ("fast_fixmatch"
                   + ("_ema" if args.use_ema else "")
                   + ("_full" if args.full else "")
                   + (f"_cbs{args.cbs_total_steps}" if args.full and args.cbs_total_steps != 60000 else "")
                   + ("_lamlab" if args.full and args.lambda_ref == "labeled" else "")
                   + ("" if args.compile else "_eager")
                   + (f"_wf{args.widen_factor}" if args.widen_factor != 2 else "")
                   + (f"_{args.tag}" if args.tag else ""))
    path = results_path(args, method_name)

    metrics = new_metrics()
    metrics["cumulative_gflops"], metrics["u_batch_size"] = [], []
    pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)
    pseudo_state = (pseudo_labels, confidences, pseudo_labels.clone(), confidences.clone(),
                    torch.tensor([label for _, label, _ in unlabeled_ds]))

    # FlexMatch CPL state, used only with --full.
    selected_label = torch.full((len(unlabeled_ds),), -1, dtype=torch.long, device=device)
    classwise_acc = torch.zeros(num_classes, dtype=torch.float32, device=device)

    losses, mask_ratios, cumulative_gflops = [], [], 0.0
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

        # --- Curriculum Batch Size: truncate the full-size unlabeled batch to what the schedule
        # prescribes for this iteration. This is the whole method. ---
        u_bs = min(cbs_unlabeled_batch_size(step, args.cbs_total_steps, max_u_batch, args.alpha,
                                            args.cbs_min_batch), x_u.shape[0])
        x_u, idx = x_u[:u_bs], idx[:u_bs]
        cumulative_gflops += step_gflops(u_bs)

        labeled_transform = strong_transform if args.full else weak_transform
        if args.optimized:
            x_l = labeled_transform(x_l).to(device, non_blocking=True, memory_format=torch.channels_last)
            y_l = y_l.to(device, non_blocking=True)
            x_u_w = weak_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
            x_u_s = strong_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_l, y_l = labeled_transform(x_l).to(device), y_l.to(device)
            x_u_w, x_u_s = weak_transform(x_u).to(device), strong_transform(x_u).to(device)
        idx_device = idx.to(device)

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16) if args.optimized else nullcontext():
                logits_u_w = model(x_u_w)
            probs_u_w = F.softmax(logits_u_w.float(), dim=1)
            max_prob, pseudo = torch.max(probs_u_w, dim=1)
            if args.full:
                # FlexMatch Curriculum Pseudo Labeling, convex mapping, with threshold warm-up.
                acc_per_sample = classwise_acc[pseudo]
                mask = max_prob.ge(args.tau * (acc_per_sample / (2.0 - acc_per_sample))).float()
                select = max_prob.ge(args.tau)
                if select.any():
                    selected_label[idx_device[select]] = pseudo[select]
                counts = torch.bincount(selected_label + 1, minlength=num_classes + 1)
                if counts.max().item() < selected_label.shape[0]:
                    classwise_acc = counts[1:].float() / max(counts.max().item(), 1)
            else:
                mask = max_prob.ge(args.tau).float()
            mask_ratios.append(mask.mean().item())

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16) if args.optimized else nullcontext():
            n_l = x_l.shape[0]
            all_logits = model(torch.cat([x_l, x_u_s], dim=0))
            loss_supervised = F.cross_entropy(all_logits[:n_l], y_l)
            loss_consistency = (mask * F.cross_entropy(all_logits[n_l:], pseudo, reduction="none")).mean()
            if args.full:
                lambda_den = batch_size_l if args.lambda_ref == "labeled" else max_u_batch
                loss_consistency = loss_consistency * (u_bs / lambda_den)
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
                                    start_time, losses, mask_ratios, pseudo_state,
                                    extra={"cumulative_gflops": cumulative_gflops, "u_batch_size": u_bs})
            losses, mask_ratios = [], []
            if stop:
                break
        elif args.verbose:
            print(f"Step {step + 1}/{args.max_steps}, u_bs: {u_bs}, Loss: {loss.item():.4f}, "
                  f"Sup: {loss_supervised.item():.4f}, Cons: {loss_consistency.item():.4f}, "
                  f"Mask Ratio: {mask_ratios[-1]:.4f}", end="\r", flush=True)


if __name__ == "__main__":
    run_fast_fixmatch()
