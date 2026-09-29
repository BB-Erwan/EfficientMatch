# Results

Every number in the paper is read from a file in this directory. Nothing here is generated at
analysis time: each file is the raw output of one training run, written as it progresses.

## Layout

```
results/<dataset>-labeled-<n>-seed-<seed>/<method>_metrics.json
```

`<method>` is built by the training script from the options it was given, so the file name records
the variant that produced it:

| Fragment | Meaning |
|---|---|
| `_ema` | accuracy measured on the EMA of the weights (the default, and what the paper reports) |
| `_wf4` | WideResNet-28-4 instead of the default WRN-28-2 (used for CIFAR-100) |
| `_mu<N>` | unlabeled:labeled ratio other than the method's own default |
| `_mixw<W>` | Mixup loss weight other than 1 (EfficientMatch only) |
| `_freematch` | EfficientMatch with FreeMatch's self-adaptive thresholding instead of a fixed tau |
| `_noclamp` | adaptive threshold left unclamped on SVHN |
| `_unlimited` | run with no accuracy target and no time budget, to the full 2^20-iteration horizon |
| `_<tag>` | free label given with `--tag`, last in the name; use one to keep your own runs from overwriting the paper's files |

Each file holds one array per metric, with one entry per evaluation: `step`, `test_acc`, `test_f1`,
`train_loss`, `time_elapsed` (seconds since the first training step), `pl_quality` (fraction of
retained pseudo-labels that are correct), `mask_ratio`, and pseudo-label transition counts.

## Which file backs which table

Rather than listing the runs here, ask the entry point, which knows the mapping and checks it
against what is on disk:

```bash
python run_experiment.py --list     # experiments, the paper tables they feed, how many are done
python run_experiment.py --check    # every run the paper reports, and whether its file exists
```

At the time of writing, `--check` reports 118/118. Those 118 files cover the main comparison
(Tables 2, 3, 12, 13), the mu, lambda_mix and mixing-target ablations (Tables 5 to 8), the matched-mu
study (Table 10), the FreeMatch-thresholding variant (Table 4 and Appendix A.5) and the
extended-horizon diagnostic (Table 11).

## Runs that are not in the paper

The directory also holds runs that were measured while designing the protocol and are not reported:
other seeds (0, 42, 666, 1401, 99998, 99999), label budgets dropped after pilot runs (SVHN with 40
and 1000 labels, CIFAR-100 with 250 and 5000), and superseded variants, recognisable by their names:

| Name | Why it is not reported |
|---|---|
| `fixmatch_ema_orig` | first FixMatch run on CIFAR-10/250 seed 2312; re-run because its wall-clock time was inflated by a machine slowdown |
| `fixmatch_ema_wf2_obsolete` | CIFAR-100 run left at the default width before the switch to WRN-28-4 |
| `mixmatch_ema_target60_wf4` | CIFAR-100 run stopped at a 60% target, a threshold the paper does not use |
| `efficientmatch_flex`, `efficientmatch_hard_flex` | FlexMatch-style class-adaptive thresholding, superseded by the FreeMatch variant of Table 4 |

They are kept because they document how the protocol was arrived at. They are reproducible with the
same scripts, but `run_experiment.py` does not list them: it only covers what the paper reports.

## Timing

`time_elapsed` is raw wall-clock time and includes periodic evaluation. The paper reports *corrected*
time, which subtracts the measured cost of those evaluations (553.7 ms per pass for WRN-28-2,
1499.1 ms for WRN-28-4); `python scripts/run_analysis.py compare` prints both.

Wall-clock time is the metric most exposed to the machine: the same run repeated on the same
machine has occasionally been 20% to 200% slower, without any concurrent job. Step counts and FLOPs
do not depend on how fast the machine runs, and are unaffected. Runs whose timing was visibly
inflated were replayed, and the ones that were not are flagged in `docs/`.

No run is bit-reproducible either. The seed fixes the labeled/unlabeled split and the first training
step exactly, but bfloat16 autocast, TF32 matrix products and cuDNN's choice of convolution
algorithm make two runs of the same command drift apart from the second step on. Repeating a run
therefore gives a slightly different curve and a slightly different step count at the target, not
the identical file.
