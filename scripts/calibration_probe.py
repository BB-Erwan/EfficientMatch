"""Calibration and in-between behaviour of the checkpoints saved by --diagnostics (Mixup probe,
CIFAR-10/250). For every checkpoint (5k, 10k, 20k iterations), on the CIFAR-10 test set:
accuracy, mean confidence, expected calibration error (15 bins) and NLL; on the 50k training
images: the fraction predicted with confidence >= 0.95 and, among those, the fraction that is
wrong (confident errors, the pseudo-labels FixMatch-style masking would admit); and, following
Zhang et al. (2018, Fig. 2), the "miss" rate at x = lam*x_i + (1-lam)*x_j for random test pairs
of different classes: a prediction is a miss if it is neither y_i nor y_j.
Writes results/mixup_probe_calibration.json.
"""
import glob
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.data import DATA_ROOT, DATASETS  # noqa: E402
from models import build_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spec = DATASETS["cifar10"]
norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])])


def load(train):
    ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=train, download=False, transform=norm)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor(ds.targets)
    return x, y


@torch.no_grad()
def probs(model, x, bs=1000):
    return torch.cat([F.softmax(model(x[i:i + bs].to(device)).float(), 1).cpu() for i in range(0, len(x), bs)])


def ece(p, y, bins=15):
    conf, pred = p.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return total


def main():
    x_te, y_te = load(False)
    x_tr, y_tr = load(True)
    g = torch.Generator().manual_seed(0)
    i, j = torch.randint(0, len(x_te), (4000,), generator=g), torch.randint(0, len(x_te), (4000,), generator=g)
    keep = y_te[i] != y_te[j]
    i, j = i[keep][:2000], j[keep][:2000]
    lams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "cifar10-labeled-250-seed-*", "*_step*.pt"))):
        seed = re.search(r"seed-(\d+)", path).group(1)
        name, step = re.match(r"(.*)_step(\d+)\.pt", os.path.basename(path)).groups()
        model = build_model("wideresnet", 10, 2, 28).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        p_te = probs(model, x_te)
        p_tr = probs(model, x_tr)
        conf_tr, pred_tr = p_tr.max(1)
        admitted = conf_tr >= 0.95
        miss = {}
        for lam in lams:
            p = probs(model, lam * x_te[i] + (1 - lam) * x_te[j])
            pred = p.argmax(1)
            miss[str(lam)] = ((pred != y_te[i]) & (pred != y_te[j])).float().mean().item()
        row = {
            "test_acc": (p_te.argmax(1) == y_te).float().mean().item(),
            "test_mean_conf": p_te.max(1).values.mean().item(),
            "test_ece": ece(p_te, y_te),
            "test_nll": F.nll_loss(torch.log(p_te.clamp_min(1e-12)), y_te).item(),
            "train_admitted_frac": admitted.float().mean().item(),
            "train_admitted_error": ((pred_tr != y_tr) & admitted).float().sum().item() / max(admitted.sum().item(), 1),
            "miss_rate": miss,
        }
        out.setdefault(name, {}).setdefault(seed, {})[step] = row
        print(f"{name} seed {seed} step {step}: acc {row['test_acc']:.3f} conf {row['test_mean_conf']:.3f} "
              f"ece {row['test_ece']:.3f} admitted {row['train_admitted_frac']:.3f} "
              f"adm_err {row['train_admitted_error']:.3f} miss@0.5 {miss['0.5']:.3f}", flush=True)
    with open(os.path.join(ROOT, "results", "mixup_probe_calibration.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
