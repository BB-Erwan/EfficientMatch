"""Compare the wall-clock time per training iteration of FixMatch (fixed unlabeled batch size)
against Fast FixMatch's Curriculum Batch Size (CBS) variant, where the unlabeled batch size is
NOT fixed but grows over training according to the "B-EXP" schedule from the Fast FixMatch paper
(Chen, Dun & Kyrillidis, arXiv:2309.03469):

    B(t) = u * (1 - (1 - t/T) / ((1 - alpha) + alpha * (1 - t/T))),  alpha = 0.7

where u is the nominal (maximum) unlabeled batch size (batch_size_l * mu), t is the current step
and T the nominal curriculum horizon (--total_steps). The schedule ramps smoothly from ~0 at t=0
up to u at t=T, floored at --cbs_min_batch. Since the batch size (and therefore the per-iteration
cost) changes throughout training, "per iteration" isn't a single number for Fast FixMatch the way
it is for FixMatch -- this script times it at several points along the curriculum (--fractions of
T) plus a plain average, rather than reporting one misleading figure.

Both variants run the exact same FixMatch training step (weak-view pseudo-labeling with confidence
masking, fused labeled+strong-unlabeled forward pass, single backward) on WideResNet-28-{widen_factor}
-- the only difference being the unlabeled batch size fed into that step. Deliberately UNOPTIMIZED:
no torch.compile, no channels_last, no autocast (plain FP32), matching the --no-optimized code path
used elsewhere in this repo (fixmatch.py) -- this isolates the batch-size effect on compute time
from the rest of the optimization stack.

Uses a small in-memory pool of real CIFAR-10 images (resampled with replacement each step) rather
than a DataLoader, since a DataLoader's fixed batch_size can't easily be changed every iteration --
fine for a pure compute-time benchmark, which doesn't care about epoch coverage or accuracy.

Usage:
    python benchmark_fast_fixmatch.py
    python benchmark_fast_fixmatch.py --mu 7 --total_steps 20000 --bench_iters 50
    python benchmark_fast_fixmatch.py --fractions 0.0 0.25 0.5 0.75 1.0
"""
import argparse
import statistics
import time

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2

from models import build_model


def cbs_unlabeled_batch_size(step, total_steps, max_batch, alpha, min_batch):
    """B-EXP Curriculum Batch Size schedule (see module docstring)."""
    frac_left = 1.0 - step / total_steps
    denom = (1 - alpha) + alpha * frac_left
    raw = max_batch * (1.0 - frac_left / denom)
    return int(max(min_batch, min(max_batch, round(raw))))


def load_cifar10_pool(pool_size, seed):
    ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:pool_size]
    to_tensor = transforms.ToTensor()
    imgs = torch.stack([to_tensor(ds[i][0]) for i in idx])
    labels = torch.tensor([ds[i][1] for i in idx])
    return imgs, labels


def sample_batch(pool, bs):
    idx = torch.randint(0, pool.shape[0], (bs,))
    return pool[idx]


def build_step_fn(weak_transform, strong_transform, device, tau, labeled_imgs, labeled_labels,
                   unlabeled_imgs, batch_size_l):
    def run_one_step(model, optimizer, u_batch_size):
        x_l = sample_batch(labeled_imgs, batch_size_l)
        y_l = labeled_labels[torch.randint(0, labeled_labels.shape[0], (batch_size_l,))]
        x_u = sample_batch(unlabeled_imgs, u_batch_size)

        x_l_t = weak_transform(x_l).to(device)
        y_l_t = y_l.to(device)
        x_u_w = weak_transform(x_u).to(device)
        x_u_s = strong_transform(x_u).to(device)

        with torch.no_grad():
            logits_u_w = model(x_u_w)
        probs_u_w = F.softmax(logits_u_w, dim=1)
        max_prob, pseudo = torch.max(probs_u_w, dim=1)
        mask = max_prob.ge(tau).float()

        optimizer.zero_grad(set_to_none=True)
        n_l = x_l_t.shape[0]
        all_logits = model(torch.cat([x_l_t, x_u_s], dim=0))
        logits_l = all_logits[:n_l]
        logits_u_s = all_logits[n_l:]

        loss = F.cross_entropy(logits_l, y_l_t) + (
            mask * F.cross_entropy(logits_u_s, pseudo, reduction="none")
        ).mean()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    return run_one_step


def time_batch_size(run_one_step, model, optimizer, u_batch_size, warmup_iters, bench_iters):
    for _ in range(warmup_iters):
        run_one_step(model, optimizer, u_batch_size)
    times = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        run_one_step(model, optimizer, u_batch_size)
        times.append(time.perf_counter() - t0)
    mean_s = statistics.mean(times)
    std_s = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_s, std_s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch_size_l", type=int, default=64)
    parser.add_argument("--mu", type=int, default=7, help="Unlabeled:labeled ratio -> FixMatch's fixed unlabeled batch size = batch_size_l * mu.")
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.7, help="Fast FixMatch CBS shape parameter (0.7 in the paper).")
    parser.add_argument("--cbs_min_batch", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=20000, help="Nominal CBS curriculum horizon T.")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                         help="t/T points along the CBS curriculum to benchmark.")
    parser.add_argument("--pool_size", type=int, default=4096, help="Unlabeled image pool kept in memory for sampling.")
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--bench_iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])
    weak_transform = v2.Compose([
        v2.ToImage(), v2.RandomHorizontalFlip(), v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std),
    ])
    strong_transform = v2.Compose([
        v2.RandAugment(num_ops=3, magnitude=5),
        v2.ToImage(), v2.RandomHorizontalFlip(), v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std),
    ])

    print("Chargement du pool CIFAR-10 (images réelles, tirées avec remise)...")
    labeled_imgs, labeled_labels = load_cifar10_pool(args.batch_size_l * 4, seed=args.seed)
    unlabeled_imgs, _ = load_cifar10_pool(args.pool_size, seed=args.seed + 1)

    max_u_batch = args.batch_size_l * args.mu
    run_one_step = build_step_fn(weak_transform, strong_transform, device, args.tau,
                                  labeled_imgs, labeled_labels, unlabeled_imgs, args.batch_size_l)

    def fresh_model_and_optimizer():
        model = build_model("wideresnet", 10, args.widen_factor, 28).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4, nesterov=True)
        return model, optimizer

    print(f"\nbatch_size_l={args.batch_size_l}, mu={args.mu} -> batch non-etiquete fixe (FixMatch) = {max_u_batch}")
    print(f"Code SANS optimisation (pas de torch.compile / channels_last / autocast), "
          f"{args.warmup_iters} it. de chauffe + {args.bench_iters} it. chronometrees.\n")

    model, optimizer = fresh_model_and_optimizer()
    fx_mean, fx_std = time_batch_size(run_one_step, model, optimizer, max_u_batch, args.warmup_iters, args.bench_iters)
    print(f"FixMatch (batch fixe = {max_u_batch:4d})            : {fx_mean * 1000:7.1f} +/- {fx_std * 1000:.1f} ms/it")

    print(f"\nFast FixMatch (Curriculum Batch Size, B-EXP, alpha={args.alpha}, T={args.total_steps}):")
    fast_means = []
    for frac in args.fractions:
        t = frac * args.total_steps
        bs = cbs_unlabeled_batch_size(t, args.total_steps, max_u_batch, args.alpha, args.cbs_min_batch)
        model, optimizer = fresh_model_and_optimizer()  # fresh model per point: independent, no state drift between timings
        mean_s, std_s = time_batch_size(run_one_step, model, optimizer, bs, args.warmup_iters, args.bench_iters)
        fast_means.append(mean_s)
        speedup = fx_mean / mean_s
        print(f"  t/T={frac:4.2f}  batch_u={bs:4d}  : {mean_s * 1000:7.1f} +/- {std_s * 1000:.1f} ms/it   (x{speedup:.2f} vs FixMatch)")

    avg_fast = statistics.mean(fast_means)
    print(f"\nMoyenne Fast FixMatch sur les {len(args.fractions)} points echantillonnes : "
          f"{avg_fast * 1000:.1f} ms/it  (x{fx_mean / avg_fast:.2f} plus rapide que FixMatch en moyenne)")


if __name__ == "__main__":
    main()
