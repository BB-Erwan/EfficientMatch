"""EfficientMatch -- the method of the paper (section 3).

FixMatch's supervised and consistency losses, plus a third term: a Mixup channel filtered by the
confidence mask the consistency loss has already computed. Nothing new is thresholded.

At each iteration, with a labeled batch X and two views of an unlabeled batch, weak U_w and strong U_s:

  L_s    cross-entropy on the labeled batch
  L_u    cross-entropy between the pseudo-label of U_w and the prediction on U_s, masked by
         confidence >= tau
  L_mix  every sample of X + U_w is mixed with a shuffled partner from the same pool, and the
         mixed sample contributes only if its *partner* passed the mask

  L = L_s + L_u + lambda_mix * L_mix

The whole contribution is `loss_mixup` below. Filtering after mixing rather than before it is what
separates this from a naive combination of MixMatch and FixMatch: the signal injected into L_mix is
never built from a partner the model deems unreliable.

Two thresholding variants of the paper are reachable from here: --adaptive_threshold for
FlexMatch's per-class curriculum, and --freematch_threshold for FreeMatch's self-adaptive rule
(Table 4). Both replace the fixed tau; neither is the default.

--mix_mask selects which confidence mask gates L_mix: 'partner' (default,
the method), 'none' (unfiltered Mixup), 'anchor', 'both' (anchor AND partner), or 'premix'
(unreliable unlabeled samples are removed from the pool before mixing; partners are drawn among
reliable samples only). Model calls and batch sizes are identical across variants, so the FLOPs
per iteration are the same.

--mix_input extra_strong is a control for the Mixup channel: the third forward pass gets a fresh
strong augmentation of each pool sample, with no mixing, trained against the sample's own target
(true label or hard pseudo-label) and weighted by --mix_mask (use 'none' to match unfiltered Mixup).
It asks whether the Mixup channel helps as mixing or merely as one more augmented view; the model
calls and FLOPs are unchanged.

    python efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80
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
from common.data import build_loaders, build_transforms, class_counts, image_size, load_datasets, split_labeled_unlabeled
from common.recording import evaluate_and_log, new_metrics, results_path, track_pseudo_labels
from common.setup import build_model_and_optimizer, enable_runtime_optimizations, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--mu", type=int, default=3,
                    help="Unlabeled:labeled batch size ratio. 3 is EfficientMatch's default, chosen in "
                         "Appendix A.1 as the value minimising total FLOPs among those that converge reliably.")
parser.add_argument("--tau", type=float, default=0.95,
                    help="Confidence threshold. The same mask gates the consistency loss and the Mixup channel.")
parser.add_argument("--mixup_weight", type=float, default=1.0,
                    help="lambda_mix, the weight of the Mixup loss (Appendix A.2, Table 7).")
parser.add_argument("--alpha", type=float, default=0.75,
                    help="Beta concentration for the mixing coefficient, drawn then folded to lambda >= 0.5 "
                         "so the anchor sample always dominates its partner.")
parser.add_argument("--mixing_target", type=str, default="semi_soft", choices=["hard", "semi_soft", "soft"],
                    help="What label the Mixup channel is trained against (Appendix A.3, Table 8). "
                         "'semi_soft', the default, mixes the two one-hot labels; 'hard' collapses that "
                         "mixture back to a single class, treating the mixed sample as a plain augmentation; "
                         "'soft' mixes with the partner's full softmax distribution, a noisier target. "
                         "Results are written as efficientmatch_hard_* and efficientmatch_soft_*.")
parser.add_argument("--mix_mask", type=str, default="partner",
                    choices=["partner", "none", "anchor", "both", "premix"],
                    help="Which confidence mask gates the Mixup loss. 'partner' "
                         "(default) is the method of the paper. Non-default values are written as "
                         "efficientmatch_*_mixmask_<value>.")
parser.add_argument("--mix_input", type=str, default="mixup", choices=["mixup", "extra_strong", "extra_weak"],
                    help="'mixup' (default) mixes each pool sample with its partner. 'extra_strong' "
                         "replaces the mix by a fresh strong augmentation of the sample itself, with its "
                         "own target (control). 'extra_weak' does the same with a fresh weak augmentation, "
                         "the kind of view Mixup mixes. Non-default results are written as *_extrastrong "
                         "or *_extraweak.")
parser.add_argument("--partner_class", type=str, default="any", choices=["any", "same", "different"],
                    help="Only with --mix_mask none. 'same' pairs each pool sample with a partner of the same "
                         "class (true label for labeled samples, pseudo-label for unlabeled ones), "
                         "'different' with a partner of another class. Partners are then drawn with "
                         "replacement; a sample with no valid partner is mixed with itself. Written as "
                         "*_partners_<value>.")
parser.add_argument("--partner_pool", type=str, default="all", choices=["all", "labeled", "unlabeled"],
                    help="Only with --mix_mask none. Restrict partners to the labeled or to the unlabeled part "
                         "of the pool (drawn with replacement). Written as *_pool_<value>.")
parser.add_argument("--diagnostics", type=str2bool, default=False,
                    help="Also record, at every evaluation, the coverage and precision of the stored "
                         "pseudo-labels, their per-class recall, the precision of the admitted pseudo-labels "
                         "and the accuracy of all weak-view pseudo-labels over the interval; and save the "
                         "evaluated model at 5k, 10k and 20k iterations under checkpoints/. Written as *_diag.")
parser.add_argument("--T", type=float, default=0.5, help="Sharpening temperature (unused at the defaults).")
parser.add_argument("--adaptive_threshold", type=str2bool, default=False,
                    help="Replace the fixed tau with FlexMatch's per-class curriculum thresholding.")
parser.add_argument("--thresh_warmup", type=str2bool, default=True,
                    help="Only with --adaptive_threshold: count still-unassigned samples in the per-class "
                         "normalisation during warmup.")
parser.add_argument("--freematch_threshold", type=str2bool, default=False,
                    help="Replace the fixed tau with FreeMatch's self-adaptive threshold, as RegMixMatch "
                         "implements it: threshold = time_p * p_model[class] / max(p_model), with time_p and "
                         "p_model tracked by EMA (0.999) over the weak-view predictions and initialised "
                         "uniformly at 1/num_classes. This is the variant of Table 4; results are written as "
                         "efficientmatch_freematch_*. Mutually exclusive with --adaptive_threshold.")
parser.add_argument("--freematch_svhn_clamp", type=str2bool, default=True,
                    help="Only with --freematch_threshold: clamp the adaptive threshold to [0.9, 0.95] on "
                         "SVHN, as the reference FreeMatch and RegMixMatch code does. Without it the variant "
                         "never converges on SVHN (Appendix A.5); results are then written as "
                         "efficientmatch_freematch_noclamp_*.")
args = parser.parse_args()


def mixing_partners_and_weights(mix_mask, anchor_masks):
    """Partner index for every sample of the X + U_w pool, and the per-sample weight of L_mix.

    anchor_masks holds 1 for labeled samples and the confidence mask for unlabeled ones. All variants
    keep the pool size fixed and the loss normalised over the whole pool (.mean()), so they differ
    only in which mixed samples count:
      partner  partners from a uniform shuffle; weight = partner's mask (the method of the paper)
      none     partners from a uniform shuffle; weight = 1 (unfiltered Mixup)
      anchor   partners from a uniform shuffle; weight = anchor's own mask
      both     partners from a uniform shuffle; weight = anchor's mask * partner's mask
      premix   unreliable samples are removed before mixing: partners are a shuffle of the reliable
               samples only, and unreliable anchors get weight 0. Implemented with static shapes.
    """
    n = anchor_masks.size(0)
    device = anchor_masks.device
    if mix_mask == "premix":
        reliable = anchor_masks > 0
        # Random keys, reliable samples first: the first r entries of `order` are a uniform
        # shuffle of the reliable indices. The k-th reliable anchor is paired with the k-th of them.
        keys = torch.rand(n, device=device) + (~reliable).float() * 2.0
        order = torch.argsort(keys)
        rank = (torch.cumsum(reliable.long(), dim=0) - 1).clamp(min=0)
        partner = torch.where(reliable, order[rank], torch.arange(n, device=device))
        return partner, anchor_masks
    partner = torch.randperm(n, device=device)
    partner_masks = anchor_masks[partner]
    if mix_mask == "partner":
        return partner, partner_masks
    if mix_mask == "none":
        return partner, torch.ones_like(anchor_masks)
    if mix_mask == "anchor":
        return partner, anchor_masks
    if mix_mask == "both":
        return partner, anchor_masks * partner_masks
    raise ValueError(f"Unknown mix_mask: {mix_mask}")


def constrained_partners(classes, n_l, partner_class, partner_pool):
    """Partner index for every sample of the X + U_w pool under a class or pool constraint.

    classes holds the true label of the n_l labeled samples followed by the pseudo-label of the
    unlabeled ones. Each sample draws one partner uniformly among the allowed ones (with
    replacement, never itself); a sample with no allowed partner is paired with itself.
    """
    n = classes.size(0)
    idx = torch.arange(n, device=classes.device)
    allowed = idx.view(-1, 1) != idx.view(1, -1)
    if partner_class == "same":
        allowed &= classes.view(-1, 1) == classes.view(1, -1)
    elif partner_class == "different":
        allowed &= classes.view(-1, 1) != classes.view(1, -1)
    if partner_pool == "labeled":
        allowed &= (idx < n_l).view(1, -1)
    elif partner_pool == "unlabeled":
        allowed &= (idx >= n_l).view(1, -1)
    scores = torch.rand(n, n, device=classes.device).masked_fill(~allowed, -1.0)
    best, partner = scores.max(dim=1)
    return torch.where(best >= 0, partner, idx)


def run_efficientmatch():
    enable_runtime_optimizations()
    device = select_device()

    num_classes, mean, std, train_ds, test_ds = load_datasets(args.dataset)
    logger.info(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Seed once, here: the split below and the weight initialisation further down both draw from
    # this point, and their order is what makes a seed reproduce a run.
    torch.manual_seed(args.seed)
    labeled_ds, unlabeled_ds = split_labeled_unlabeled(train_ds, num_classes, args.num_labeled)
    logger.info(f"Labeled class distribution: {class_counts(labeled_ds, num_classes)}")

    norm_transform, weak_transform, strong_transform = build_transforms(mean, std, image_size(args.dataset))
    batch_size_l = 64
    labeled_loader, unlabeled_loader, test_loader, unlabeled_ds = build_loaders(
        labeled_ds, unlabeled_ds, test_ds, norm_transform, batch_size_l, args.mu, args.optimized)

    model, base_model, eval_model, ema, optimizer, scheduler = build_model_and_optimizer(
        args, num_classes, device)

    if args.mix_input != "mixup" and args.mix_mask not in ("none", "anchor"):
        raise ValueError("--mix_input extra_strong/extra_weak has no partner: use --mix_mask none or anchor.")
    constrained = args.partner_class != "any" or args.partner_pool != "all"
    if constrained and (args.mix_mask != "none" or args.mix_input != "mixup"):
        raise ValueError("--partner_class / --partner_pool require --mix_mask none and --mix_input mixup.")
    if args.adaptive_threshold and args.freematch_threshold:
        raise ValueError("--adaptive_threshold (FlexMatch) and --freematch_threshold are mutually exclusive.")
    variant = {"hard": "efficientmatch_hard", "soft": "efficientmatch_soft"}.get(args.mixing_target,
                                                                                "efficientmatch")
    method_name = (("efficientmatch_freematch" if args.freematch_threshold else variant)
                   + ("_noclamp" if args.freematch_threshold and not args.freematch_svhn_clamp else "")
                   + ("_flex" if args.adaptive_threshold else "")
                   + ("_ema" if args.use_ema else "")
                   + (f"_mu{args.mu}" if args.mu != 3 else "")
                   + (f"_mixw{args.mixup_weight}" if args.mixup_weight != 1.0 else "")
                   + (f"_mixmask_{args.mix_mask}" if args.mix_mask != "partner" else "")
                   + ("_extrastrong" if args.mix_input == "extra_strong" else "")
                   + ("_extraweak" if args.mix_input == "extra_weak" else "")
                   + (f"_partners_{args.partner_class}" if args.partner_class != "any" else "")
                   + (f"_pool_{args.partner_pool}" if args.partner_pool != "all" else "")
                   + ("_diag" if args.diagnostics else "")
                   + (f"_wf{args.widen_factor}" if args.widen_factor != 2 else "")
                   + (f"_{args.tag}" if args.tag else ""))
    path = results_path(args, method_name)

    beta_dist = torch.distributions.Beta(torch.tensor(args.alpha, device=device, dtype=torch.float32),
                                         torch.tensor(args.alpha, device=device, dtype=torch.float32))

    # State of the two optional thresholding rules. Unused at the defaults.
    selected_label = torch.full((len(unlabeled_ds),), -1, dtype=torch.long, device=device)
    classwise_acc = torch.zeros(num_classes, dtype=torch.float32, device=device)
    time_p = torch.tensor(1.0 / num_classes, device=device)
    p_model = torch.full((num_classes,), 1.0 / num_classes, device=device)

    metrics = new_metrics()
    pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)
    pseudo_state = (pseudo_labels, confidences, pseudo_labels.clone(), confidences.clone(),
                    torch.tensor([label for _, label, _ in unlabeled_ds]))

    diag_keys = ["pl_coverage", "pl_precision", "pl_class_recall", "admitted_precision", "weak_pl_acc"]
    if args.diagnostics:
        for key in diag_keys:
            metrics[key] = []
        # Running sums on the GPU (no per-step sync): admitted & correct, admitted, correct, seen.
        diag_sums = torch.zeros(4, device=device)
        ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints",
                                os.path.basename(os.path.dirname(path)))
        os.makedirs(ckpt_dir, exist_ok=True)

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

        x_l_raw = x_l  # kept for the --mix_input extra_strong control
        # Both views are built here, from the same raw batch, then moved to the GPU.
        if args.optimized:
            x_l = weak_transform(x_l).to(device, non_blocking=True, memory_format=torch.channels_last)
            y_l = y_l.to(device, non_blocking=True)
            x_u_w = weak_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
            x_u_s = strong_transform(x_u).to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x_l, y_l = weak_transform(x_l).to(device), y_l.to(device)
            x_u_w, x_u_s = weak_transform(x_u).to(device), strong_transform(x_u).to(device)
        idx_device = idx.to(device, non_blocking=args.optimized)

        # --- Pseudo-labels, without gradient: one forward pass on the weak view. ---
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_u_w = model(x_u_w)
            probs_u_w = F.softmax(logits_u_w.float(), dim=1)
            max_prob, pseudo = torch.max(probs_u_w, dim=1)

            if args.adaptive_threshold:
                acc_per_sample = classwise_acc[pseudo]
                mask = max_prob.ge(args.tau * (acc_per_sample / (2.0 - acc_per_sample))).float()
                select = max_prob.ge(args.tau)
                if select.any():
                    selected_label[idx_device[select]] = pseudo[select]
                # Bin 0 counts the still-unselected (-1) samples, bins 1.. the per-class counts.
                # thresh_warmup relies on bin 0 dominating early training to keep classwise_acc near
                # 0, and the mask permissive, until enough samples are confidently pseudo-labeled --
                # so it must not be filtered out beforehand.
                counts = torch.bincount(selected_label + 1, minlength=num_classes + 1)
                if counts.max().item() < selected_label.shape[0]:
                    if args.thresh_warmup:
                        denom = max(counts.max().item(), 1)
                    else:
                        without_unselected = counts.clone()
                        without_unselected[0] = 0
                        denom = max(without_unselected.max().item(), 1)
                    classwise_acc = counts[1:].float() / denom
            elif args.freematch_threshold:
                time_p = time_p * 0.999 + max_prob.mean() * 0.001
                p_model = p_model * 0.999 + probs_u_w.mean(dim=0) * 0.001
                threshold = time_p * (p_model / p_model.max())[pseudo]
                if args.dataset == "svhn" and args.freematch_svhn_clamp:
                    threshold = torch.clamp(threshold, min=0.9, max=0.95)
                mask = max_prob.ge(threshold).float()
            else:
                mask = max_prob.ge(args.tau).float()

            mask_ratios.append(mask.mean().item())

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16) if args.optimized else nullcontext():
            n_l = x_l.shape[0]

            # --- Supervised and consistency losses, in one fused forward pass. ---
            all_logits = model(torch.cat([x_l, x_u_s], dim=0))
            loss_supervised = F.cross_entropy(all_logits[:n_l], y_l)
            loss_consistency = (mask * F.cross_entropy(all_logits[n_l:], pseudo, reduction="none")).mean()

            # --- Confidence-filtered Mixup: the contribution of this method. ---
            # Labeled samples enter with their true label and a mask of 1, unlabeled ones with their
            # pseudo-label and the mask above. The pool is shuffled to draw a partner for each
            # sample, and each sample is mixed with its partner.
            one_hot_l = F.one_hot(y_l, num_classes=num_classes).float()
            one_hot_u = F.one_hot(pseudo, num_classes=num_classes).float()
            all_inputs = torch.cat([x_l, x_u_w], dim=0)
            all_targets = torch.cat([one_hot_l, one_hot_u], dim=0)
            anchor_masks = torch.cat([torch.ones(n_l, device=device), mask], dim=0)

            if constrained:
                partner = constrained_partners(torch.cat([y_l, pseudo]), n_l, args.partner_class,
                                               args.partner_pool)
                mix_weights = torch.ones_like(anchor_masks)
            else:
                partner, mix_weights = mixing_partners_and_weights(args.mix_mask, anchor_masks)
            all_inputs, all_targets = all_inputs[partner], all_targets[partner]

            # lambda >= 0.5 keeps the anchor dominant in the mix.
            lam_x = beta_dist.sample((n_l,))
            lam_u = beta_dist.sample((x_u_w.size(0),))
            lam_x = torch.maximum(lam_x, 1 - lam_x)
            lam_u = torch.maximum(lam_u, 1 - lam_u)

            mixup_x = torch.lerp(all_inputs[:n_l], x_l, lam_x.view(-1, 1, 1, 1))
            mixup_u = torch.lerp(all_inputs[n_l:], x_u_w, lam_u.view(-1, 1, 1, 1))
            # The unlabeled anchor's own target is its one-hot pseudo-label, or its full softmax
            # distribution for the noisier "soft" variant of Appendix A.3.
            anchor_u = probs_u_w if args.mixing_target == "soft" else one_hot_u
            mixup_targets_x = torch.lerp(all_targets[:n_l], one_hot_l, lam_x.view(-1, 1))
            mixup_targets_u = torch.lerp(all_targets[n_l:], anchor_u, lam_u.view(-1, 1))

            if args.mix_input == "extra_strong":
                # Control: same pass size and weights, but each sample is a fresh strong view of
                # itself, with its own target, instead of a mix with its partner.
                if args.optimized:
                    extra_l = strong_transform(x_l_raw).to(device, non_blocking=True,
                                                           memory_format=torch.channels_last)
                    extra_u = strong_transform(x_u).to(device, non_blocking=True,
                                                       memory_format=torch.channels_last)
                else:
                    extra_l, extra_u = strong_transform(x_l_raw).to(device), strong_transform(x_u).to(device)
                mixup_x, mixup_u = extra_l, extra_u
                mixup_targets_x, mixup_targets_u = one_hot_l, anchor_u
            elif args.mix_input == "extra_weak":
                # Same control with a fresh weak view: the kind of view the Mixup channel mixes.
                if args.optimized:
                    extra_l = weak_transform(x_l_raw).to(device, non_blocking=True,
                                                         memory_format=torch.channels_last)
                    extra_u = weak_transform(x_u).to(device, non_blocking=True,
                                                     memory_format=torch.channels_last)
                else:
                    extra_l, extra_u = weak_transform(x_l_raw).to(device), weak_transform(x_u).to(device)
                mixup_x, mixup_u = extra_l, extra_u
                mixup_targets_x, mixup_targets_u = one_hot_l, anchor_u

            mixup_logits = model(torch.cat([mixup_x, mixup_u], dim=0))
            mixup_targets = torch.cat([mixup_targets_x, mixup_targets_u], dim=0)
            # "hard" collapses the mixed target back to one class, which -- since lambda >= 0.5 --
            # is always the anchor's, turning the mix into a plain augmentation.
            if args.mixing_target == "hard":
                mixup_targets = mixup_targets.argmax(dim=1)

            # With the default --mix_mask partner, mix_weights is the *partner's* mask after the
            # shuffle: a mixed sample counts only if the sample it was mixed with was confident enough.
            loss_mixup = args.mixup_weight * (
                mix_weights * F.cross_entropy(mixup_logits, mixup_targets, reduction="none")
            ).mean()

            loss = loss_supervised + loss_consistency + loss_mixup
            losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(base_model)

        track_pseudo_labels(pseudo_labels, confidences, idx, mask, pseudo, max_prob)
        if args.diagnostics:
            with torch.no_grad():
                correct = (pseudo == y_u.to(device, non_blocking=True)).float()
                diag_sums += torch.stack([(mask * correct).sum(), mask.sum(), correct.sum(),
                                          torch.tensor(float(correct.numel()), device=device)])

        if (step + 1) % args.test_period == 0 or step == 0 or step == args.max_steps - 1:
            if ema is not None:
                ema.copy_to(eval_model)
            extra = None
            if args.diagnostics:
                true_labels = pseudo_state[4]
                stored = pseudo_labels != -1
                good = pseudo_labels == true_labels
                s = diag_sums.tolist()
                extra = {
                    "pl_coverage": stored.float().mean().item(),
                    "pl_precision": (good[stored].float().mean().item() if stored.any() else 0.0),
                    "pl_class_recall": [(good & (true_labels == c)).sum().item()
                                        / max((true_labels == c).sum().item(), 1)
                                        for c in range(num_classes)],
                    "admitted_precision": s[0] / s[1] if s[1] > 0 else 0.0,
                    "weak_pl_acc": s[2] / s[3] if s[3] > 0 else 0.0,
                }
                diag_sums.zero_()
            stop = evaluate_and_log(args, step, metrics, path, eval_model, test_loader, device,
                                    start_time, losses, mask_ratios, pseudo_state, extra=extra)
            if args.diagnostics and step + 1 in (5000, 10000, 20000):
                torch.save(eval_model.state_dict(),
                           os.path.join(ckpt_dir, f"{os.path.basename(path)[:-len('_metrics.json')]}"
                                                  f"_step{step + 1}.pt"))
            losses, mask_ratios = [], []
            if stop:
                break
        elif args.verbose:
            print(f"Step {step + 1}/{args.max_steps}, Loss: {loss.item():.4f}, "
                  f"Sup: {loss_supervised.item():.4f}, Cons: {loss_consistency.item():.4f}, "
                  f"Mixup: {loss_mixup.item():.4f}, Mask Ratio: {mask_ratios[-1]:.4f}",
                  end="\r", flush=True)


if __name__ == "__main__":
    run_efficientmatch()
