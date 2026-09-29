"""Metrics every method records, and the periodic evaluation that writes them out.

All of it is shared because all of it is the same: accuracy and F1 on the test set, the running
training loss, the share of pseudo-labels the confidence mask admits, and how the pseudo-labels
themselves move between two evaluations -- how many are newly assigned, corrected, or broken, and
whether rising confidence reinforces right or wrong labels.

That last family is what Figure 1 of the paper is drawn from: it separates a method whose accuracy
tracks the quality of the labels it trains on from one that outstrips it.
"""
import json
import logging
import os
import time

import numpy as np
import torch

from utils import evaluate_f1_and_accuracy

logger = logging.getLogger(__name__)

METRIC_KEYS = ["step", "train_loss", "test_f1", "test_acc", "time_elapsed", "pl_quality",
               "mask_ratio", "corrections", "new_errors", "error_reinforcement",
               "correct_reinforcement", "new_label", "new_correct", "bad_corrections", "topk_acc"]


def new_metrics():
    return {key: [] for key in METRIC_KEYS}


def results_path(args, method_name):
    """results/<dataset>-labeled-<n>-seed-<seed>/<method_name>_metrics.json, created if needed."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory = os.path.join(repo_root, "results",
                             f"{args.dataset}-labeled-{args.num_labeled}-seed-{args.seed}")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{method_name}_metrics.json")


def compute_pseudo_label_metrics(pseudo_labels, confidences, last_pseudo_labels, last_confidences,
                                 true_labels):
    """How the pseudo-labels moved since the previous evaluation: newly assigned labels,
    corrections (wrong -> right) against bad corrections (right -> wrong), and whether rising
    confidence reinforces correct or incorrect labels."""
    new_label = (pseudo_labels != last_pseudo_labels) & (last_pseudo_labels == -1)
    changed = (pseudo_labels != last_pseudo_labels) & (last_pseudo_labels != -1)
    was_good = last_pseudo_labels == true_labels
    is_good = pseudo_labels == true_labels
    more_confident = confidences > last_confidences

    return {
        "pl_quality": is_good.sum().item() / len(true_labels),
        "corrections": (changed & ~was_good & is_good).sum().item(),
        "bad_corrections": (changed & was_good & ~is_good).sum().item(),
        "new_label": new_label.sum().item(),
        "new_errors": (new_label & ~is_good).sum().item(),
        "new_correct": (new_label & is_good).sum().item(),
        "correct_reinforcement": (more_confident & is_good & was_good & ~new_label).sum().item(),
        "error_reinforcement": (more_confident & ~is_good & ~was_good & ~new_label).sum().item(),
    }


def track_pseudo_labels(pseudo_labels, confidences, indices, mask, pseudo, max_prob):
    """Remember the pseudo-label of every sample the mask admitted this iteration; the ones it
    rejected keep whatever they were last assigned. Modifies both tensors in place."""
    admitted = mask.bool().cpu()
    pseudo_labels[indices] = torch.where(admitted, pseudo.cpu(), pseudo_labels[indices])
    confidences[indices] = torch.where(admitted, max_prob.cpu(), confidences[indices])


def evaluate_and_log(args, step, metrics, path, eval_model, test_loader, device, start_time,
                     losses, mask_ratios, pseudo_state, extra=None):
    """Evaluate, append one entry to every metric, rewrite the JSON file, log a summary line, and
    return True if the run should stop -- because it reached --target_acc or ran out of
    --max_minutes.

    `extra` adds one entry to metrics keys a method records beyond the shared set -- Fast FixMatch's
    per-step batch size and cumulative FLOPs are the only case.

    `pseudo_state` is the mutable 4-tuple (pseudo_labels, confidences, last_pseudo_labels,
    last_confidences, true_labels) carried by the training loop; the two "last" tensors are
    refreshed here, since this is where one evaluation interval ends and the next begins.

    The file is rewritten at every evaluation rather than once at the end, so a run stopped halfway
    still leaves usable data -- which is how the `†` rows of the paper were measured.
    """
    pseudo_labels, confidences, last_pseudo_labels, last_confidences, true_labels = pseudo_state
    f1, acc, topk = evaluate_f1_and_accuracy(eval_model, test_loader, device, args.topk)

    metrics["step"].append(step + 1)
    metrics["test_f1"].append(f1)
    metrics["test_acc"].append(acc)
    metrics["time_elapsed"].append(time.time() - start_time)
    metrics["mask_ratio"].append(np.mean(mask_ratios))
    metrics["train_loss"].append(np.mean(losses))
    metrics["topk_acc"].append(topk)

    pl = compute_pseudo_label_metrics(pseudo_labels, confidences, last_pseudo_labels,
                                      last_confidences, true_labels)
    for key, value in pl.items():
        metrics[key].append(value)
    for key, value in (extra or {}).items():
        metrics[key].append(value)
    last_pseudo_labels.copy_(pseudo_labels)
    last_confidences.copy_(confidences)

    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=4)

    if args.verbose:
        print()   # the per-iteration line is rewritten in place; close it before logging
    topk_str = (" ".join(f"Top-{k}: {v:.4f}," for k, v in enumerate(topk, start=1))
                if args.topk > 1 else "")
    logger.info(
        f"Test F1: {f1:.4f}, Acc: {acc:.4f}, {topk_str} PL Quality: {pl['pl_quality']:.4f}, "
        f"Mask Ratio: {metrics['mask_ratio'][-1]:.4f}, Error Reinforcement: {pl['error_reinforcement']}, "
        f"Correct Reinforcement: {pl['correct_reinforcement']}, Corrections: {pl['corrections']}, "
        f"Bad Corrections: {pl['bad_corrections']}, New Errors: {pl['new_errors']}, "
        f"New Correct: {pl['new_correct']}, Loss: {metrics['train_loss'][-1]:.4f}, "
        f"Time: {metrics['time_elapsed'][-1]:.2f}s"
    )

    if args.target_acc is not None and acc >= args.target_acc:
        logger.info(f"Reached target_acc={args.target_acc:.4f} at step {step + 1} "
                    f"(acc={acc:.4f}) -- stopping early.")
        return True
    if args.max_minutes is not None and metrics["time_elapsed"][-1] >= args.max_minutes * 60:
        logger.info(f"Reached the {args.max_minutes:.0f}-minute budget at step {step + 1} "
                    f"(acc={acc:.4f}) -- stopping.")
        return True
    return False
