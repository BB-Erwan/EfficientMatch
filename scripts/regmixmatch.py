"""RegMixMatch (Tao et al.), ported from the reference implementation at
https://github.com/hhrd9/regmixmatch (file `freematch_entropy.py` / `models/freematch_entropy/`,
built on top of FreeMatch: Wang et al., ICLR 2023).

Structure: FreeMatch's self-adaptive thresholding (a global threshold `time_p` and a per-class
threshold modulator `p_model`, both EMA-tracked from the model's own weak-augmented predictions,
replacing a hand-picked fixed tau) + FreeMatch's self-adaptive fairness term (`entropy_loss`,
maximizing the cross-entropy between the modulated model/prediction class histograms to counteract
class imbalance in the pseudo-labels) + RegMixMatch's own contribution, a confidence-routed
ResizeMix: a "confident" pool (labeled data + high-confidence unlabeled pseudo-labels, threshold
`tau_m`) is self-mixed to build extra supervised signal (`mix_loss`), and low-confidence samples
are optionally patched with a small same-class region from that pool (`cam_loss`) to anchor their
own consistency target -- the paper's ablations found this second term unreliable, so it is off by
default here (`--disab_cam true`), matching the upstream README's final recommendation.

Deviation from upstream: the reference implementation seeds the EMA trackers (`time_p`, `p_model`,
`label_hist`) with a warmup pass over the *test* set. We seed them from the unlabeled weak-augmented
set instead -- our other scripts never let training touch test data before an evaluation checkpoint,
and there is no reason this one should either.
"""
import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.args import add_common_args, str2bool
from common.data import build_loaders, build_transforms, class_counts, load_datasets, split_labeled_unlabeled
from common.recording import evaluate_and_log, new_metrics, results_path
from common.setup import build_model_and_optimizer, enable_runtime_optimizations, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def rand_bbox_tao(size, tao):
    """Random box whose side is a `tao` fraction of the image (ResizeMix's random scale box)."""
    W, H = size[2], size[3]
    cut_w, cut_h = int(W * tao), int(H * tao)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def resizemix(img, gt_label, crop_ratio=0.9, scope=(0.1, 0.8), alpha=1.0):
    """Self-mixup within one pool: paste a resized center-crop of a shuffled partner
    into a random box of each sample (Qin et al., ResizeMix)."""
    rand_index = torch.randperm(img.size(0), device=img.device)
    img = img.clone()
    img_resize = img[rand_index]
    shuffled_gt = gt_label[rand_index]
    _, _, h, w = img.size()

    tao = np.sqrt(np.random.beta(alpha, alpha))
    tao = scope[0] + tao * (scope[1] - scope[0])
    bbx1, bby1, bbx2, bby2 = rand_bbox_tao(img.size(), tao)

    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    img_resize = img_resize[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
    img_resize = F.interpolate(img_resize, (bby2 - bby1, bbx2 - bbx1), mode="nearest")

    img[:, :, bby1:bby2, bbx1:bbx2] = img_resize
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    mixed_y = lam * gt_label + (1 - lam) * shuffled_gt
    return img, mixed_y


def resizemix_cam(x, y_soft, pool_x, pool_y, crop_ratio=0.9, scope=(0.1, 0.8), alpha=16.0):
    """Paste a small same-class patch from the confident pool onto each uncertain sample,
    keeping its consistency target anchored to its own predicted class."""
    x = x.clone()
    y_soft = y_soft.clone()
    pool_resize = pool_x.clone()
    _, _, h, w = x.size()
    tao = np.sqrt(np.random.beta(alpha, alpha))
    tao = scope[0] + tao * (scope[1] - scope[0])
    bbx1, bby1, bbx2, bby2 = rand_bbox_tao(x.size(), tao)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    pool_resize = pool_resize[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
    pool_resize = F.interpolate(pool_resize, (bby2 - bby1, bbx2 - bbx1), mode="nearest")

    labels = y_soft.argmax(dim=-1)
    pool_labels = pool_y.argmax(dim=-1)
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        pool_idx = (pool_labels == c).nonzero(as_tuple=True)[0]
        if pool_idx.numel() == 0:
            continue
        pick = pool_idx[torch.randint(0, pool_idx.numel(), (idx.numel(),), device=x.device)]
        x[idx, :, bby1:bby2, bbx1:bbx2] = pool_resize[pick]
        y_soft[idx] = lam * y_soft[idx] + (1 - lam) * pool_y[pick]
    return x, y_soft


def resizemix_l(x, y_soft, pool_x, pool_y, crop_ratio=0.9, scope=(0.1, 0.8), alpha=16.0):
    """Same as resizemix_cam but the donor patch is drawn at random, ignoring class."""
    x = x.clone()
    pool_resize = pool_x.clone()
    _, _, h, w = x.size()
    tao = np.sqrt(np.random.beta(alpha, alpha))
    tao = scope[0] + tao * (scope[1] - scope[0])
    bbx1, bby1, bbx2, bby2 = rand_bbox_tao(x.size(), tao)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    pool_resize = pool_resize[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
    pool_resize = F.interpolate(pool_resize, (bby2 - bby1, bbx2 - bbx1), mode="nearest")

    pick = torch.randint(0, pool_x.size(0), (x.size(0),), device=x.device)
    x = x.clone()
    x[:, :, bby1:bby2, bbx1:bbx2] = pool_resize[pick]
    mixed_y = lam * y_soft + (1 - lam) * pool_y[pick]
    return x, mixed_y


def entropy_loss(mask, logits_s, p_model, label_hist):
    """FreeMatch's self-adaptive fairness term: MAXIMIZES the cross-entropy between the
    rarity-modulated model distribution and the rarity-modulated prediction histogram of the
    (masked) strongly-augmented batch -- pushes the pseudo-label distribution away from
    collapsing onto a few classes. Note the missing minus sign is intentional: minimizing this
    term (as added to total_loss) maximizes that cross-entropy."""
    logits_s = logits_s[mask]
    if logits_s.shape[0] == 0:
        return logits_s.new_zeros(())
    prob_s = logits_s.softmax(dim=-1)
    pred_label_s = prob_s.argmax(dim=-1)
    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(prob_s.dtype)
    hist_s = hist_s / hist_s.sum()

    scaler_model = torch.nan_to_num(1.0 / label_hist, nan=0.0, posinf=0.0, neginf=0.0)
    mod_p_model = p_model * scaler_model
    mod_p_model = mod_p_model / mod_p_model.sum()

    scaler_s = torch.nan_to_num(1.0 / hist_s, nan=0.0, posinf=0.0, neginf=0.0)
    mod_mean_s = prob_s.mean(dim=0) * scaler_s
    mod_mean_s = mod_mean_s / mod_mean_s.sum()

    return (mod_p_model * torch.log(mod_mean_s + 1e-12)).sum()


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--mu", type=int, default=7, help="Unlabeled:labeled batch size ratio.")
parser.add_argument("--tau_m", type=float, default=0.999, help="Confidence threshold defining the 'confident' pool used to build the ResizeMix supervision (separate from the adaptive consistency threshold, which has no fixed tau in this method).")
parser.add_argument("--lambda_u", type=float, default=1.0, help="Weight of the FreeMatch consistency loss.")
parser.add_argument("--lambda_e", type=float, default=0.01, help="Weight of the self-adaptive fairness (entropy) loss.")
parser.add_argument("--hard_label", type=str2bool, default=True, help="Use the argmax pseudo-label (vs. a temperature-sharpened soft target) for the consistency loss.")
parser.add_argument("--T", type=float, default=0.5, help="Sharpening temperature, only used when --hard_label is false.")
parser.add_argument("--svhn_clamp", type=str2bool, default=True, help="Clamp the FreeMatch adaptive threshold to [0.9, 0.95] on SVHN, as the upstream RegMixMatch/FreeMatch code does. Set to false to use the raw adaptive threshold on SVHN too (results are then saved as regmixmatch_noclamp_*).")
parser.add_argument("--disab_cam", type=str2bool, default=True, help="Disable the confidence-routed patch loss (cam_loss) on low-confidence samples. The upstream authors found it unreliable and recommend leaving it disabled.")
parser.add_argument("--alpha_h", type=float, default=1.0, help="Beta-distribution concentration for the confident-pool ResizeMix box (larger = smaller/rarer boxes).")
parser.add_argument("--alpha_l", type=float, default=16.0, help="Beta-distribution concentration for the uncertain-sample patch box.")
parser.add_argument("--warmup_steps", type=int, default=2048, help="Purely-supervised steps run before the main loop, used only to seed the adaptive-threshold EMA trackers. Not counted towards --max_steps.")
parser.add_argument("--static_shapes", type=str2bool, default=False, help="Mix over the full labeled+unlabeled population every step and zero out non-confident contributions via a multiplicative weight, instead of slicing out a confidence-filtered subset (whose size changes every step). Mathematically equivalent, but keeps every model() call at a fixed shape, which is required for torch.compile(mode='reduce-overhead') to be safe here -- the default (filtered) path crashes under it on shape-varying CUDA graph replay. Automatically enables that compile mode when true.")
args = parser.parse_args()


def run_regmixmatch():
    enable_runtime_optimizations()
    device = select_device()

    num_classes, mean, std, train_ds, test_ds = load_datasets(args.dataset)
    logger.info(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    torch.manual_seed(args.seed)
    labeled_ds, unlabeled_ds = split_labeled_unlabeled(train_ds, num_classes, args.num_labeled)
    logger.info(f"Labeled class distribution: {class_counts(labeled_ds, num_classes)}")

    norm_transform, weak_transform, strong_transform = build_transforms(mean, std)
    optimized = args.optimized
    batch_size_l = 64
    mu = args.mu
    labeled_loader, unlabeled_loader, test_loader, unlabeled_ds = build_loaders(
        labeled_ds, unlabeled_ds, test_ds, norm_transform, batch_size_l, mu, optimized)

    # torch.compile(mode="reduce-overhead") uses CUDA graphs, which require every model() call to
    # keep the same input shape. By default the confident and uncertain pool sizes feeding the
    # second model() call change almost every step -- they depend on how many unlabeled samples
    # clear tau_m -- which crashes under that mode (assert_size_stride mismatches in
    # convolution_backward when a graph captured for one shape is replayed with another).
    # --static_shapes removes that variation at the source, so compilation is enabled only with it.
    model, base_model, eval_model, ema, optimizer, scheduler = build_model_and_optimizer(
        args, num_classes, device, compile_model=args.static_shapes)

    max_steps = args.max_steps
    total_steps = args.total_steps

    static_shapes = args.static_shapes
    lambda_u = args.lambda_u
    lambda_e = args.lambda_e
    tau_m = args.tau_m
    hard_label = args.hard_label
    T = args.T
    disab_cam = args.disab_cam
    alpha_h = args.alpha_h
    alpha_l = args.alpha_l
    confident_pool_full = (args.num_labeled / num_classes) >= 100  # rich-label regime: mix confident pool + labeled data

    method_name = "regmixmatch" + ("_noclamp" if not args.svhn_clamp else "") + (f"_mu{args.mu}" if args.mu != 7 else "") + ("_ema" if args.use_ema else "") + (f"_wf{args.widen_factor}" if args.widen_factor != 2 else "") + (f"_{args.tag}" if args.tag else "")

    path = results_path(args, method_name)

    metrics = new_metrics()
    pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)
    pseudo_state = (pseudo_labels, confidences, pseudo_labels.clone(), confidences.clone(),
                    torch.tensor([label for _, label, _ in unlabeled_ds]))

    test_period = args.test_period
    verbose = args.verbose

    use_cuda_autocast = optimized and device.type == "cuda"
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    # ── Warmup: purely-supervised steps, used only to seed the adaptive-threshold EMA trackers ──
    logger.info(f"Warmup: {args.warmup_steps} supervised-only steps to initialize the adaptive threshold.")
    model.train()
    for _ in range(args.warmup_steps):
        try:
            x_l, y_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x_l, y_l = next(labeled_iter)
        x_l = weak_transform(x_l).to(device, non_blocking=True)
        y_l = y_l.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda_autocast else nullcontext():
            loss_warmup = F.cross_entropy(model(x_l), y_l)
        loss_warmup.backward()
        optimizer.step()

    # Seed time_p / p_model / label_hist from the unlabeled weak-augmented distribution
    # (upstream seeds these from the *test* set instead; we avoid touching test data here).
    model.eval()
    probs_acc = []
    with torch.no_grad():
        for x_u, _, _ in unlabeled_loader:
            x_u = weak_transform(x_u).to(device, non_blocking=True)
            probs_acc.append(F.softmax(model(x_u).float(), dim=1).cpu())
            if sum(p.shape[0] for p in probs_acc) >= 4096:
                break
    probs_acc = torch.cat(probs_acc)
    max_probs_seed, max_idx_seed = torch.max(probs_acc, dim=-1)
    time_p = max_probs_seed.mean().to(device)
    p_model = probs_acc.mean(dim=0).to(device)
    label_hist = (torch.bincount(max_idx_seed, minlength=num_classes).float() / max_idx_seed.shape[0]).to(device)
    model.train()

    start_time = time.time()
    mask_ratio = []
    losses = []

    for step in range(max_steps):
        model.train()

        try:
            x_l, y_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x_l, y_l = next(labeled_iter)
        try:
            x_u, _, idx = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_u, _, idx = next(unlabeled_iter)
        idx_cpu = idx.cpu()

        if optimized:
            cf = torch.channels_last
            x_l_in = weak_transform(x_l).to(device, non_blocking=True, memory_format=cf)
            y_l = y_l.to(device, non_blocking=True)
            x_u_w = weak_transform(x_u).to(device, non_blocking=True, memory_format=cf)
            x_u_s = strong_transform(x_u).to(device, non_blocking=True, memory_format=cf)
        else:
            x_l_in = weak_transform(x_l).to(device)
            y_l = y_l.to(device)
            x_u_w = weak_transform(x_u).to(device)
            x_u_s = strong_transform(x_u).to(device)

        n_l = x_l_in.shape[0]
        n_u = x_u_w.shape[0]

        optimizer.zero_grad(set_to_none=True)
        train_ctx = autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda_autocast else nullcontext()
        with train_ctx:
            all_logits = model(torch.cat([x_l_in, x_u_w, x_u_s], dim=0))
            logits_l = all_logits[:n_l]
            logits_w = all_logits[n_l:n_l + n_u]
            logits_s = all_logits[n_l + n_u:]

            sup_loss = F.cross_entropy(logits_l, y_l)

            # ── FreeMatch self-adaptive threshold: update EMA trackers, then derive the mask ──
            with torch.no_grad():
                prob_w = F.softmax(logits_w.float(), dim=-1)
                max_probs, pseudo = torch.max(prob_w, dim=-1)
                time_p = time_p * 0.999 + max_probs.mean() * 0.001
                p_model = p_model * 0.999 + prob_w.mean(dim=0) * 0.001
                hist_now = torch.bincount(pseudo, minlength=num_classes).to(p_model.dtype)
                label_hist = label_hist * 0.999 + (hist_now / hist_now.sum()) * 0.001

                p_model_cutoff = p_model / p_model.max()
                threshold = time_p * p_model_cutoff[pseudo]
                if args.dataset == "svhn" and args.svhn_clamp:
                    threshold = torch.clamp(threshold, min=0.9, max=0.95)
                mask = max_probs.ge(threshold)
                valid = max_probs >= tau_m  # the "confident" pool feeding the ResizeMix supervision

            if hard_label:
                ce_per_sample = F.cross_entropy(logits_s, pseudo, reduction="none")
            else:
                soft_pseudo = F.softmax(logits_w.detach() / T, dim=-1)
                ce_per_sample = -(soft_pseudo * F.log_softmax(logits_s, dim=-1)).sum(dim=-1)
            unsup_loss = (ce_per_sample * mask.float()).mean()

            # ── Confidence-routed ResizeMix (RegMixMatch's own contribution) ──────────────
            num_ulb = n_u

            if static_shapes:
                # Mix over the FULL labeled+unlabeled population (fixed shape every step),
                # then zero out non-confident contributions via a multiplicative weight
                # instead of slicing them out beforehand -- see --static_shapes help text.
                # Trade-off: the ResizeMix donor pool can now include samples the filtered
                # path would have excluded (e.g. labeled data as a donor even when
                # confident_pool_full is false); they never contribute to the loss unless
                # they themselves pass the threshold, so this only affects which pixels a
                # masked-out row's patch is drawn from, never what gets optimized.
                conf_labels_full = torch.cat([
                    F.one_hot(y_l, num_classes).float(),
                    F.one_hot(pseudo, num_classes).float(),
                ], dim=0)
                conf_data_full = torch.cat([x_l_in, x_u_s], dim=0)
                if confident_pool_full:
                    mix_weight = torch.cat([torch.ones(n_l, device=device), valid.float()])
                else:
                    mix_weight = torch.cat([torch.zeros(n_l, device=device), valid.float()])

                mixed_x, mixed_y = resizemix(conf_data_full, conf_labels_full, alpha=alpha_h)

                if disab_cam:
                    logits_mix = model(mixed_x)
                    mix_loss_per_sample = (F.log_softmax(logits_mix, dim=-1) * -mixed_y).sum(dim=-1)
                    mix_loss = (mix_loss_per_sample * mix_weight).sum() / num_ulb
                    cam_loss = mixed_x.new_zeros(())
                else:
                    if np.random.rand() < 0.5:
                        mixed_x_sc, mixed_y_sc = resizemix_cam(x_u_s, prob_w.clone(), conf_data_full, conf_labels_full, alpha=alpha_l)
                    else:
                        mixed_x_sc, mixed_y_sc = resizemix_l(x_u_s, prob_w.clone(), conf_data_full, conf_labels_full, alpha=alpha_l)
                    n_mix = mixed_x.shape[0]
                    logits_mix_all = model(torch.cat([mixed_x, mixed_x_sc], dim=0))
                    logits_mix_cert = logits_mix_all[:n_mix]
                    logits_mix_uncert = logits_mix_all[n_mix:]
                    mix_loss_per_sample = (F.log_softmax(logits_mix_cert, dim=-1) * -mixed_y).sum(dim=-1)
                    mix_loss = (mix_loss_per_sample * mix_weight).sum() / num_ulb
                    uncert_weight = (~valid).float() * (max_probs ** 3)
                    cam_loss = (F.mse_loss(F.softmax(logits_mix_uncert, dim=-1), mixed_y_sc, reduction="none").mean(dim=-1) * uncert_weight).mean()
            else:
                conf_labels = torch.cat([
                    F.one_hot(y_l, num_classes).float(),
                    F.one_hot(pseudo[valid], num_classes).float(),
                ], dim=0)
                conf_data = torch.cat([x_l_in, x_u_s[valid]], dim=0)

                if confident_pool_full:
                    mixed_x, mixed_y = resizemix(conf_data, conf_labels, alpha=alpha_h)
                else:
                    mixed_x, mixed_y = resizemix(
                        x_u_s[valid], F.one_hot(pseudo[valid], num_classes).float(), alpha=alpha_h
                    )

                if disab_cam:
                    logits_mix = model(mixed_x)
                    mix_loss = (F.log_softmax(logits_mix, dim=-1) * -mixed_y).sum(dim=-1).sum() / num_ulb
                    cam_loss = mixed_x.new_zeros(())
                else:
                    x_u_sc = x_u_s[~valid]
                    y_sc = prob_w[~valid].clone()
                    if np.random.rand() < 0.5:
                        mixed_x_sc, mixed_y_sc = resizemix_cam(x_u_sc, y_sc, conf_data, conf_labels, alpha=alpha_l)
                    else:
                        mixed_x_sc, mixed_y_sc = resizemix_l(x_u_sc, y_sc, conf_data, conf_labels, alpha=alpha_l)
                    num_cert = mixed_x.shape[0]
                    logits_mix_all = model(torch.cat([mixed_x, mixed_x_sc], dim=0))
                    logits_mix_cert = logits_mix_all[:num_cert]
                    logits_mix_uncert = logits_mix_all[num_cert:]
                    mix_loss = (F.log_softmax(logits_mix_cert, dim=-1) * -mixed_y).sum(dim=-1).sum() / num_ulb
                    uncert_weight = max_probs[~valid] ** 3
                    cam_loss = (F.mse_loss(F.softmax(logits_mix_uncert, dim=-1), mixed_y_sc, reduction="none").mean(dim=-1) * uncert_weight).mean()

            ent_loss = entropy_loss(mask, logits_s, p_model, label_hist)

            srm_loss = lambda_u * unsup_loss + mix_loss
            loss = sup_loss + lambda_e * ent_loss + srm_loss + cam_loss
            losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(base_model)

        mask_ratio.append(mask.float().mean().item())
        mask_cpu = mask.cpu()
        pseudo_cpu = pseudo.detach().cpu()
        max_probs_cpu = max_probs.detach().cpu()
        pseudo_labels[idx_cpu] = torch.where(mask_cpu, pseudo_cpu, pseudo_labels[idx_cpu])
        confidences[idx_cpu] = torch.where(mask_cpu, max_probs_cpu, confidences[idx_cpu])

        if (step + 1) % test_period == 0 or step == 0 or step == max_steps - 1:
            if ema is not None:
                ema.copy_to(eval_model)
            stop = evaluate_and_log(args, step, metrics, path, eval_model, test_loader, device,
                                    start_time, losses, mask_ratio, pseudo_state)
            losses, mask_ratio = [], []
            if stop:
                break
        elif verbose:
            print(
                f"Step {step + 1}/{max_steps}, Loss: {loss.item():.4f}, Sup: {sup_loss.item():.4f}, "
                f"Unsup: {unsup_loss.item():.4f}, Mix: {mix_loss.item():.4f}, Ent: {ent_loss.item():.4f}, "
                f"Mask Ratio: {mask_ratio[-1]:.4f}",
                end="\r",
                flush=True,
            )


if __name__ == "__main__":
    run_regmixmatch()
