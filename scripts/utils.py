import math

import torch
from sklearn.metrics import accuracy_score, f1_score


class FixMatchCosineLR:
    """lr(k) = lr0 * cos(7*pi*k / (16*K)), schedule cosine recalé standard FixMatch/EfficientMatch."""

    def __init__(self, optimizer, total_steps, base_lr=None):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.base_lr = base_lr if base_lr is not None else optimizer.param_groups[0]["lr"]
        self.last_step = -1
        self.step()

    def step(self):
        self.last_step += 1
        new_lr = self.base_lr * math.cos(7 * math.pi * self.last_step / (16 * self.total_steps))
        new_lr = max(new_lr, 0.0)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr


def build_lr_scheduler(optimizer, total_steps, schedule="fixmatch_cosine", base_lr=None):
    """schedule: "fixmatch_cosine" (default, FixMatch/EfficientMatch rescaled cosine) or
    "cosine_annealing" (torch.optim.lr_scheduler.CosineAnnealingLR)."""
    if schedule == "fixmatch_cosine":
        return FixMatchCosineLR(optimizer, total_steps, base_lr=base_lr)
    elif schedule == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    raise ValueError(f"Unknown lr_schedule: {schedule}")


@torch.no_grad()
def evaluate_f1_and_accuracy(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc
