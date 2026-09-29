# Implementation notes

Details of the code that a reader reproducing or extending the paper should know about, because
they are not visible from the paper alone. None of them changes the comparison between methods
except where stated.

## All methods

**Augmentations are drawn once per batch.** The weak and strong augmentations are torchvision v2
transforms applied to the whole batch tensor, on the CPU, just before it is moved to the GPU. Given
a batch, these transforms draw their random parameters once: every image in it receives the same
flip, the same crop and the same RandAugment operations. FixMatch, as published, draws them per
image. All five methods are treated identically, so the comparison is unaffected, but absolute
accuracies may differ from those reported with per-image augmentation. The notebooks expose the
per-image variant as `PER_SAMPLE_AUGMENT` (`scripts/common/data.py`).

**Evaluation period.** The code evaluates every 500 iterations (`--test_period`); the paper states
512.

**Runs are not bit-reproducible.** The seed fixes the labeled/unlabeled split, the initial weights
and the first training step, but bfloat16 autocast, TF32 matrix products and cuDNN's benchmark mode
(which picks convolution algorithms by timing them) make two runs of the same command drift apart
from the second iteration on. See `results/README.md`.

**Compilation.** EfficientMatch, FixMatch, FlexMatch and MixMatch run under
`torch.compile(mode="reduce-overhead")`. RegMixMatch's main runs do not, because the size of its
confident pool changes at every iteration and CUDA graphs require fixed shapes; its matched-mu runs
(Table 10) use `--static_shapes` and are compiled (see below).

## MixMatch

**The supervised term uses the unmixed label.** `L_x` is the cross-entropy of the *mixed* labeled
images against their *original* label (`scripts/mixmatch.py`, `loss_l`), whereas MixMatch as
published uses the mixed target. That target, `mixup_targets_x`, is computed but not used. Since
the mixing coefficient is folded to λ ≥ 0.5, the original label is always the dominant one in the
mix. This term is also computed outside autocast, on the bfloat16 logits.
`notebooks/mixmatch.ipynb` shows how to switch to the published version.

## RegMixMatch

**The warmup is not counted.** Before training proper, RegMixMatch runs 2,048 supervised-only
iterations on the labeled batch (`--warmup_steps`), then seeds the moving averages of its adaptive
threshold from about 4,096 unlabeled images. This happens before the clock starts: it counts
neither in the reported time, nor in the number of iterations, nor in the FLOPs, and the
learning-rate schedule and the EMA do not advance during it. Measured on the GPU of the paper:

| | Time | FLOPs |
|---|---:|---:|
| WRN-28-2 (SVHN, CIFAR-10) | 19.5 s | 0.17 PFLOPs |
| WRN-28-4 (CIFAR-100) | 28.3 s | 0.67 PFLOPs |

Added to RegMixMatch's results, this is under 1% of its time and FLOPs to target on CIFAR-10 and
CIFAR-100, and about 2 to 4% on SVHN; no ranking of Tables 2 and 3 on time or FLOPs changes.
Counted as iterations, the warmup adds 2,048 to RegMixMatch's iteration counts, although each of
these iterations processes only the 64 labeled images, against 960 images (labeled batch plus two
views of 448 unlabeled ones) in the main forward pass of a regular iteration. Counted this way,
RegMixMatch would no longer converge in the fewest iterations on SVHN: 7,548 / 6,548 / 8,048
iterations on seeds 2312 / 0308 / 2701 (the 5,500 / 4,500 / 6,000 of Table 2, plus 2,048),
against 7,000 / 6,500 / 8,000 for EfficientMatch and 8,000 / 6,500 / 8,000 for FixMatch. On the three other
configurations it stays well ahead in iterations.

**Departures from the reference implementation** (https://github.com/hhrd9/regmixmatch):

- The reference code rebuilds the confident pool by slicing the batch, so its size changes at
  every iteration. `--static_shapes` instead mixes the whole batch and zeroes the non-confident
  contributions with a mask: what is optimised is the same, and only the images that can donate a
  ResizeMix patch differ. The main runs use the slicing path; the matched-mu runs of Table 10 use
  the mask.
- The threshold's moving averages are seeded from the unlabeled set rather than from the test set,
  so that training never touches test data.
- The class-aware ResizeMix branch (`cam_loss`) is disabled by default (`--disab_cam True`), as the
  upstream authors recommend, and no run of the paper enables it.
- ResizeMix draws its box with NumPy's global random generator, which the scripts do not seed.
