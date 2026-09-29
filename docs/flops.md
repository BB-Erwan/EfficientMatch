# FLOPs and evaluation cost

Backs **Appendix B** of the paper, and supplies the per-iteration constants that turn a step count
into the FLOP column of every other table.

```bash
python scripts/flops_analysis.py --methods efficientmatch fixmatch flexmatch mixmatch regmixmatch --widen-factor 2
python scripts/measure_eval_time.py          # cost of one evaluation pass
```

## FLOPs per iteration

Measured with `torch.utils.flop_counter.FlopCounterMode` on dummy tensors of the right shape: no
dataset, no dataloader, no training. Each `<method>_iter_flops` function reproduces exactly the
sequence of model calls the corresponding training script makes, so the count is independent of the
data, of `--optimized`, of mixed precision and of `torch.compile`.

The labeled batch is 64 everywhere; the unlabeled batch is `mu` times that, and `mu` differs by
method, which is where most of the spread comes from.

**WideResNet-28-2 (CIFAR-10, SVHN)**

| Method | mu | Unlabeled batch | GFLOPs / iteration |
|---|---:|---:|---:|
| MixMatch | 1 | 64 | 301.64 |
| EfficientMatch | 3 | 192 | 740.35 |
| FixMatch | 7 | 448 | 850.10 |
| FlexMatch | 7 | 448 | 850.10 |
| RegMixMatch | 7 | 448 | 1891.86 |
| EfficientMatch, mu = 1 | 1 | 64 | 356.46 |
| EfficientMatch, mu = 5 | 5 | 320 | 1124.25 |
| EfficientMatch, mu = 7 | 7 | 448 | 1508.14 |

**WideResNet-28-4 (CIFAR-100)**

| Method | mu | Unlabeled batch | GFLOPs / iteration |
|---|---:|---:|---:|
| MixMatch | 1 | 64 | 1190.43 |
| EfficientMatch | 3 | 192 | 2921.93 |
| FixMatch | 7 | 448 | 3354.88 |
| FlexMatch | 7 | 448 | 3354.88 |
| RegMixMatch | 7 | 448 | 7467.01 |

At equal mu, EfficientMatch is more expensive per iteration than FixMatch, since it adds one forward
pass over the mixed batch; at the mu each method actually uses, it is cheaper than every method
except MixMatch. The mixing-target variants of Appendix A.3 share EfficientMatch's count exactly:
they change only the loss formula, which is an elementwise and reduction operation that adds no
convolution or matrix product.

CIFAR-100 was originally intended to run at WRN-28-8, but that width exhausts the 8 GB of the GPU
used here with several methods, so WRN-28-4 is the architecture for every CIFAR-100 result reported.
FLOP counts are architecture-specific and must never be carried across widths.

## Cost of the FreeMatch-thresholding variant

The adaptive threshold of Table 4 adds elementwise work that `FlopCounterMode` does not attribute
any convolution or matrix product to. Counting it by hand — one mean, two EMA updates, one max, one
division, one comparison per unlabeled sample, one FLOP per operation, indexing free — gives:

| Configuration | FLOPs / iteration | Overhead vs EfficientMatch |
|---|---:|---:|
| CIFAR-10, 10 classes | 740 351 510 836 | +2 356 (3.2e-9 relative) |
| SVHN, 10 classes, clamp active | 740 351 511 220 | +2 740 (3.7e-9 relative) |
| CIFAR-100, 100 classes (WRN-28-4) | 2 921 930 903 158 | +20 086 (6.9e-9 relative) |

Over a whole run this amounts to at most 3.2e8 FLOPs, which disappears in the rounding of the
PFLOP-scale tables. The variant's speed advantage comes entirely from admitting more pseudo-labels
per iteration, not from a different per-iteration cost.

## Cost of one evaluation

Every method calls the same `evaluate_f1_and_accuracy()` on the same model type and test loader
every `test_period` steps, so the cost of an evaluation depends only on the architecture and the
test set, not on the training method. It was measured twice: in isolation, and in real conditions
with `ghost_method.py`, which runs the full pipeline — dataloaders, augmentations, EMA,
`torch.compile` — with a training step that does nothing, so that only evaluation touches the GPU.

| Architecture | Test set | Isolated | In situ | Gap |
|---|---|---:|---:|---:|
| WRN-28-2 | CIFAR-10, 10 000 images | 544.7 ms | 553.7 ms | +1.6% |
| WRN-28-4 | CIFAR-100, 10 000 images | 1 492.8 ms | 1 499.1 ms | +0.4% |

The two measurements agreeing within 2% is what makes the isolated figure usable to correct run
times. The corrected time reported everywhere else is

```
corrected = raw − (number of evaluations × cost of one evaluation)
```

with 553.7 ms for WRN-28-2 and 1499.1 ms for WRN-28-4.

For the run with the most evaluations on each dataset — the worst case — this overhead is 5.9% of
the total on CIFAR-10 (MixMatch, 250 labels, seed 0308: 1767 evaluations over 276.6 min) and stays
under 1.5% on CIFAR-100. For speed comparisons between methods on one configuration it is therefore
not usually a confounding factor, except for very short runs evaluating very often, where it can
reach about 10% of the measured time.
