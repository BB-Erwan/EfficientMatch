"""Measure how much wall-clock time a single evaluation pass (the test_period step) actually
costs, independently of any SSL method. Every training script in this repo calls the same
evaluate_f1_and_accuracy() on the same kind of model/test_loader at every test_period steps, so
the evaluation overhead only depends on (dataset, model architecture, topk) -- not on which
method is training. This script benchmarks that cost directly, so it can be subtracted from a
run's total time_elapsed to recover the "pure training time" when comparing methods that don't
use the same test_period (or the same number of evaluations).

Not meant to be run automatically as part of a sweep -- run manually when needed:
    python scripts/measure_eval_time.py --dataset svhn --topk 1
    python scripts/measure_eval_time.py --dataset cifar100 --widen_factor 8 --num_repeats 20
"""

import argparse
import logging
import sys
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append("..")

from datasets_utils import TransformedDataset
from models import build_model
from utils import evaluate_f1_and_accuracy
from ema import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DATASET_CONFIG = {
    "cifar10": dict(num_classes=10, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    "cifar100": dict(num_classes=100, mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
    "svhn": dict(num_classes=10, mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
}


def build_test_loader(dataset, mean, std, optimized):
    norm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == "cifar100":
        test_ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
    elif dataset == "svhn":
        test_ds = torchvision.datasets.SVHN(root="./data", split="test", download=True)
    else:
        test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    test_ds = TransformedDataset(test_ds, norm_transform)

    dl_kwargs = dict(num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True) if optimized else {}
    return DataLoader(test_ds, batch_size=256, shuffle=False, **dl_kwargs), len(test_ds)


def measure(dataset, model_name, widen_factor, depth, topk, num_repeats, optimized, device):
    cfg = DATASET_CONFIG[dataset]
    test_loader, n_test = build_test_loader(dataset, cfg["mean"], cfg["std"], optimized)

    train_model = build_model(model_name, cfg["num_classes"], widen_factor, depth).to(device)
    eval_model = build_model(model_name, cfg["num_classes"], widen_factor, depth).to(device)
    if optimized:
        train_model = train_model.to(memory_format=torch.channels_last)
        eval_model = eval_model.to(memory_format=torch.channels_last)
    eval_model.eval()
    # Every real training script calls ema.copy_to(eval_model) immediately before each
    # evaluation -- verified in situ (scripts/ghost_method.py) to add a measurable ~5% on top
    # of the bare evaluate_f1_and_accuracy() call, so it must be included here too.
    ema = EMA(train_model, 0.999)

    # Warmup: first call pays for cuDNN autotuning / lazy CUDA init, not representative.
    for _ in range(2):
        ema.copy_to(eval_model)
        evaluate_f1_and_accuracy(eval_model, test_loader, device, topk)

    times = []
    for _ in range(num_repeats):
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        ema.copy_to(eval_model)
        evaluate_f1_and_accuracy(eval_model, test_loader, device, topk)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.perf_counter() - t0)

    times = torch.tensor(times)
    return {
        "dataset": dataset,
        "model": model_name,
        "widen_factor": widen_factor,
        "depth": depth,
        "n_test": n_test,
        "topk": topk,
        "mean_s": times.mean().item(),
        "std_s": times.std().item(),
        "min_s": times.min().item(),
        "max_s": times.max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, nargs="+", default=["cifar10", "cifar100", "svhn"], choices=list(DATASET_CONFIG))
    parser.add_argument("--model", type=str, default="wideresnet", choices=["wideresnet", "resnet18"])
    parser.add_argument("--widen_factor", type=int, default=2, help="Use 8 to match the cifar100 configs run in this repo.")
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--num_repeats", type=int, default=15, help="Number of timed evaluation passes to average over.")
    parser.add_argument("--optimized", type=lambda s: s.lower() != "false", default=True, help="Match --optimized as used by the training scripts (channels_last + persistent dataloader workers).")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    rows = []
    for dataset in args.dataset:
        logger.info(f"Measuring eval time for {dataset} (model={args.model}, widen_factor={args.widen_factor}, depth={args.depth}, topk={args.topk})...")
        row = measure(dataset, args.model, args.widen_factor, args.depth, args.topk, args.num_repeats, args.optimized, device)
        rows.append(row)

    print(f"\n{'dataset':<10}{'model':<12}{'wf':>4}{'depth':>7}{'n_test':>8}{'topk':>6}{'mean_ms':>10}{'std_ms':>9}{'min_ms':>9}{'max_ms':>9}")
    for r in rows:
        print(
            f"{r['dataset']:<10}{r['model']:<12}{r['widen_factor']:>4}{r['depth']:>7}{r['n_test']:>8}{r['topk']:>6}"
            f"{r['mean_s']*1000:>10.1f}{r['std_s']*1000:>9.1f}{r['min_s']*1000:>9.1f}{r['max_s']*1000:>9.1f}"
        )

    print(
        "\nTo recover a run's pure training time: "
        "training_time = time_elapsed - n_evals * mean_eval_time_s "
        "(n_evals = number of rows in that run's metrics JSON, or roughly max_steps / test_period)."
    )


if __name__ == "__main__":
    main()
