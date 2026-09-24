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
from torchvision.transforms import v2

sys.path.append("..")  # add parent directory to path for imports

from datasets_utils import TransformedDataset, TransformedDatasetWithIndex
from models import build_model
from utils import evaluate_f1_and_accuracy, build_lr_scheduler
from ema import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_pseudo_label_metrics(pseudo_labels, confidences, last_pseudo_labels, last_confidences, true_labels):
    """Track how pseudo-labels evolve between two evaluations: new labels assigned,
    corrections (wrong -> right) vs bad corrections (right -> wrong), and whether
    confidence increases reinforce correct or incorrect labels."""
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


# ── RegMixMatch-specific building blocks (ported from freematch_utils.py) ──────


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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
parser.add_argument("--widen_factor", type=int, default=2, help="WideResNet-28-{widen_factor}. Papers use 2 for CIFAR-10 and 8 for CIFAR-100.")
parser.add_argument("--model", type=str, default="wideresnet", choices=["wideresnet", "resnet18"], help="Backbone architecture.")
parser.add_argument("--depth", type=int, default=28, help="WideResNet depth (e.g. 28 for WRN-28-x, 40 for WRN-40-x). Ignored for resnet18. Must satisfy (depth-4) mod 6 == 0.")
parser.add_argument("--topk", type=int, default=1, help="Also report top-k accuracy for every k from 1 to this value (e.g. --topk 3 logs Top-1, Top-2 and Top-3).")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="SGD weight decay. Papers use 5e-4 for CIFAR-10 and 1e-3 for CIFAR-100.")
parser.add_argument("--num_labeled", type=int, default=250)
parser.add_argument("--optimized", type=str2bool, default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test_period", type=int, default=500)
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
parser.add_argument("--max_steps", type=int, default=2**20, help="Number of steps actually run; the run is truncated here.")
parser.add_argument("--total_steps", type=int, default=2**20, help="Nominal horizon the cosine LR schedule decays over, independent of max_steps.")
parser.add_argument("--lr_schedule", type=str, default="fixmatch_cosine", choices=["fixmatch_cosine", "cosine_annealing"], help="LR schedule: rescaled FixMatch cosine (default) or torch's classic CosineAnnealingLR.")
parser.add_argument("--verbose", type=str2bool, default=False)
parser.add_argument("--target_acc", type=float, default=None, help="Stop the run early once test_acc reaches this value.")
parser.add_argument("--use_ema", type=str2bool, default=True, help="Evaluate an EMA of the weights instead of the raw training weights.")
parser.add_argument("--ema_decay", type=float, default=0.999)
args = parser.parse_args()


def run_regmixmatch():
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

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    weak_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    strong_transform = v2.Compose([
        v2.RandAugment(num_ops=3, magnitude=5),
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

    labeled_ds = TransformedDataset(labeled_ds, transforms.ToTensor())
    unlabeled_ds = TransformedDatasetWithIndex(unlabeled_ds, transform=transforms.ToTensor())
    test_ds = TransformedDataset(test_ds, norm_transform)

    num_workers = min(2, os.cpu_count())
    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    batch_size_l = 64
    mu = args.mu

    if optimized:
        labeled_loader = DataLoader(labeled_ds, batch_size=batch_size_l, shuffle=True, **dl_kwargs)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size_l * mu, shuffle=True, **dl_kwargs)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, **dl_kwargs)
    else:
        labeled_loader = DataLoader(labeled_ds, batch_size=batch_size_l, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size_l * mu, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_model(args.model, num_classes, args.widen_factor, args.depth).to(device)
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
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.03, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
    )
    scheduler = build_lr_scheduler(optimizer, total_steps, schedule=args.lr_schedule)

    # torch.compile(mode="reduce-overhead") uses CUDA graphs, which require every model() call
    # to keep the same input shape. By default the confident/uncertain pool sizes that feed the
    # second model() call change almost every step (they depend on how many unlabeled samples
    # clear the tau_m threshold), which crashes under that mode (assert_size_stride mismatches
    # in convolution_backward when a graph captured for one shape gets replayed with another).
    # --static_shapes fixes this at the source (see below), so only enable compile then.
    if args.static_shapes and optimized and torch.cuda.is_available() and "5060 Ti" in torch.cuda.get_device_name(0):
        try:
            import triton
            triton_available = True
        except ImportError:
            triton_available = False
        if triton_available:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile activé (mode=reduce-overhead)")

    static_shapes = args.static_shapes
    lambda_u = args.lambda_u
    lambda_e = args.lambda_e
    tau_m = args.tau_m
    hard_label = args.hard_label
    T = args.T
    disab_cam = args.disab_cam
    alpha_h = args.alpha_h
    alpha_l = args.alpha_l
    confident_pool_full = (num_labeled / num_classes) >= 100  # rich-label regime: mix confident pool + labeled data

    method_name = "regmixmatch" + ("_noclamp" if not args.svhn_clamp else "") + (f"_mu{args.mu}" if args.mu != 7 else "") + ("_ema" if args.use_ema else "") + (f"_wf{args.widen_factor}" if args.widen_factor != 2 else "")
    dataset_prefix = f"{args.dataset}-"
    name_of_experiment = f"{dataset_prefix}labeled-{num_labeled}-seed-{args.seed}"

    metrics = {
        "step": [],
        "train_loss": [],
        "test_f1": [],
        "test_acc": [],
        "time_elapsed": [],
        "pl_quality": [],
        "mask_ratio": [],
        "corrections": [],
        "new_errors": [],
        "error_reinforcement": [],
        "correct_reinforcement": [],
        "new_label": [],
        "new_correct": [],
        "bad_corrections": [],
        "topk_acc": [],
    }

    last_pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    last_confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)
    true_labels = torch.tensor([label for _, label, _ in unlabeled_ds])
    pseudo_labels = torch.full((len(unlabeled_ds),), -1, dtype=torch.long)
    confidences = torch.zeros(len(unlabeled_ds), dtype=torch.float32)

    test_period = args.test_period
    verbose = args.verbose
    target_acc = args.target_acc

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results", name_of_experiment)
    os.makedirs(results_dir, exist_ok=True)

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
            f1, acc, topk_accs = evaluate_f1_and_accuracy(eval_model, test_loader, device, args.topk)
            metrics["step"].append(step + 1)
            metrics["test_f1"].append(f1)
            metrics["test_acc"].append(acc)
            metrics["time_elapsed"].append(time.time() - start_time)
            metrics["mask_ratio"].append(float(np.mean(mask_ratio)) if mask_ratio else 0.0)
            metrics["train_loss"].append(float(np.mean(losses)) if losses else 0.0)
            metrics["topk_acc"].append(topk_accs)
            mask_ratio = []
            losses = []

            pl_metrics = compute_pseudo_label_metrics(
                pseudo_labels, confidences, last_pseudo_labels, last_confidences, true_labels
            )
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
                f"Mask Ratio: {metrics['mask_ratio'][-1]:.4f}, Error Reinforcement: {pl_metrics['error_reinforcement']}, Correct Reinforcement: {pl_metrics['correct_reinforcement']}, "
                f"Corrections: {pl_metrics['corrections']}, Bad Corrections: {pl_metrics['bad_corrections']}, New Errors: {pl_metrics['new_errors']}, New Correct: {pl_metrics['new_correct']}, Loss: {metrics['train_loss'][-1]:.4f}, "
                f"Time: {metrics['time_elapsed'][-1]:.2f}s"
            )

            if target_acc is not None and acc >= target_acc:
                logger.info(f"Reached target_acc={target_acc:.4f} at step {step + 1} (acc={acc:.4f}) — stopping early.")
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
