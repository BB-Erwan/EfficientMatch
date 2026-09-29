"""MixMatch (Berthelot et al., 2019) -- Mixup without confidence filtering.

No strong augmentation and no threshold. Instead, a guessed label is formed by averaging the
predictions over two weakly augmented views of the same unlabeled batch and sharpening the result,
then everything -- labeled and unlabeled alike -- is mixed with a shuffled partner:

  guess  = sharpen((p(y | U_w1) + p(y | U_w2)) / 2, T)
  L_x    cross-entropy on the mixed labeled part, against its mixed one-hot target
  L_u    mean squared error on the mixed unlabeled part, against its mixed guess

  L = L_x + 100 * ramp(step) * L_u,    ramp rising linearly over the first 16 000 iterations

Every pseudo-label enters the mix, reliable or not; the rising weight on L_u is the only thing
limiting their influence early on. That is what makes MixMatch reverse between label budgets: at
4000 labels on CIFAR-10 it converges faster than FixMatch and FlexMatch, at 250 labels it never
reaches the target inside the budget on any seed (see docs/main_experiment.md). EfficientMatch is
built to keep this method's benefit while filtering what enters the mix.

    python mixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80
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
from common.recording import evaluate_and_log, new_metrics, results_path
from common.setup import build_model_and_optimizer, enable_runtime_optimizations, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--T", type=float, default=0.5,
                    help="Sharpening temperature applied to the averaged guess. Lower is sharper.")
parser.add_argument("--alpha", type=float, default=0.75,
                    help="Beta concentration for the mixing coefficient, folded to lambda >= 0.5.")
parser.add_argument("--tau", type=float, default=0.95,
                    help="Unused: MixMatch applies no confidence threshold. Accepted so that every method "
                         "takes the same options.")
args = parser.parse_args()

MU = 1              # MixMatch draws as many unlabeled samples as labeled ones
LAMBDA_U = 100      # weight of the unsupervised term, reached at the end of the ramp
RAMP_STEPS = 16000  # iterations over which that weight rises linearly from 0


def run_mixmatch():
    enable_runtime_optimizations()
    device = select_device()

    num_classes, mean, std, train_ds, test_ds = load_datasets(args.dataset)
    logger.info(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    torch.manual_seed(args.seed)
    labeled_ds, unlabeled_ds = split_labeled_unlabeled(train_ds, num_classes, args.num_labeled)
    logger.info(f"Labeled class distribution: {class_counts(labeled_ds, num_classes)}")

    norm_transform, weak_transform, _ = build_transforms(mean, std)
    batch_size_l = 64
    labeled_loader, unlabeled_loader, test_loader, unlabeled_ds = build_loaders(
        labeled_ds, unlabeled_ds, test_ds, norm_transform, batch_size_l, MU, args.optimized)

    model, base_model, eval_model, ema, optimizer, scheduler = build_model_and_optimizer(
        args, num_classes, device)

    method_name = ("mixmatch"
                   + ("_ema" if args.use_ema else "")
                   + (f"_wf{args.widen_factor}" if args.widen_factor != 2 else "")
                   + (f"_{args.tag}" if args.tag else ""))
    path = results_path(args, method_name)

    beta_dist = torch.distributions.Beta(torch.tensor(args.alpha, device=device, dtype=torch.float32),
                                         torch.tensor(args.alpha, device=device, dtype=torch.float32))

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

        # Two independent weak views of the same unlabeled batch; no strong augmentation here.
        if args.optimized:
            x_l = weak_transform(x_l).to(device, non_blocking=True, memory_format=torch.channels_last)
            y_l = y_l.to(device, non_blocking=True)
            x_u_w_1 = weak_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
            x_u_w_2 = weak_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_l, y_l = weak_transform(x_l).to(device), y_l.to(device)
            x_u_w_1, x_u_w_2 = weak_transform(x_u).to(device), weak_transform(x_u).to(device)

        mix_ctx = autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()

        # --- Label guessing: average the two views, then sharpen. ---
        with mix_ctx:
            with torch.no_grad():
                x_u_pair = torch.cat([x_u_w_1, x_u_w_2], dim=0)
                all_logits = model(x_u_pair)
                probs_1 = F.softmax(all_logits[: x_u_w_1.shape[0]], dim=1)
                probs_2 = F.softmax(all_logits[x_u_w_1.shape[0]:], dim=1)
                probs_avg = (probs_1 + probs_2) / 2

                sharpened = probs_avg ** (1 / args.T)
                sharpened = sharpened / sharpened.sum(dim=1, keepdim=True)

        # --- Mixup over the whole batch, labeled and unlabeled together. ---
        with mix_ctx:
            n_l = x_l.size(0)
            y_l_onehot = F.one_hot(y_l, num_classes).to(dtype=sharpened.dtype)
            guesses = torch.cat([sharpened, sharpened], dim=0)   # one per view

            all_inputs = torch.cat([x_l, x_u_pair], dim=0)
            all_targets = torch.cat([y_l_onehot, guesses], dim=0)

            partner = torch.randperm(all_inputs.size(0), device=all_inputs.device)
            all_inputs, all_targets = all_inputs[partner], all_targets[partner]

            # lambda >= 0.5 keeps the anchor dominant in the mix.
            lam_x = beta_dist.sample((n_l,))
            lam_u = beta_dist.sample((x_u_pair.size(0),))
            lam_x = torch.maximum(lam_x, 1 - lam_x)
            lam_u = torch.maximum(lam_u, 1 - lam_u)

            mixup_x = torch.lerp(all_inputs[:n_l], x_l, lam_x.view(-1, 1, 1, 1))
            mixup_u = torch.lerp(all_inputs[n_l:], x_u_pair, lam_u.view(-1, 1, 1, 1))
            mixup_targets_x = torch.lerp(all_targets[:n_l], y_l_onehot, lam_x.view(-1, 1))
            mixup_targets_u = torch.lerp(all_targets[n_l:], guesses, lam_u.view(-1, 1))

            all_logits = model(torch.cat([mixup_x, mixup_u], dim=0))
            logits_l, logits_u = all_logits[:n_l], all_logits[n_l:]

        loss_l = F.cross_entropy(logits_l, y_l)

        optimizer.zero_grad(set_to_none=True)
        loss_u = F.mse_loss(F.softmax(logits_u.float(), dim=1), mixup_targets_u.float())
        ramp = min(1.0, step / RAMP_STEPS)
        loss = loss_l.float() + LAMBDA_U * ramp * loss_u
        loss.backward()
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(base_model)
        losses.append(loss.item())

        # Every unlabeled sample is used at every iteration, so its guess is recorded unconditionally
        # -- there is no mask to filter on, which is why the mask ratio is 1 throughout.
        idx_cpu = idx.cpu()
        pseudo_labels[idx_cpu] = torch.argmax(probs_avg, dim=1).cpu()
        confidences[idx_cpu] = torch.max(probs_avg, dim=1).values.cpu()
        mask_ratios.append(1.0)

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
                  f"Sup: {loss_l.item():.4f}, Unsup: {loss_u.item():.4f}, ramp: {ramp:.3f}",
                  end="\r", flush=True)


if __name__ == "__main__":
    run_mixmatch()
