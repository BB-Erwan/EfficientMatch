# Reproducing the paper, command by command

Every run behind a number in the paper, as a plain command line. `run_experiment.py` drives the same
list and skips what is already in `results/`, so prefer it unless you want to dispatch the runs
yourself; this file is generated from it with

```bash
python run_experiment.py --commands
```

Each command stops at its target accuracy or after two hours, whichever comes first, and writes one
JSON file under `results/`. The main sweep alone is 60 runs, so expect days rather than hours on a
single GPU. Add `--seed`, `--config` or `--method` to `run_experiment.py` to run a subset.

Figures are regenerated separately, and in seconds, since they only read the result files:

```bash
python make_figures.py
```

## The runs

```bash
# main -- Tables 2, 3, 12, 13
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/fixmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/fixmatch.py --dataset svhn --num_labeled 250 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/fixmatch.py --dataset svhn --num_labeled 250 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/flexmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/flexmatch.py --dataset svhn --num_labeled 250 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/flexmatch.py --dataset svhn --num_labeled 250 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/mixmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/mixmatch.py --dataset svhn --num_labeled 250 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/mixmatch.py --dataset svhn --num_labeled 250 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/regmixmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/regmixmatch.py --dataset svhn --num_labeled 250 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/regmixmatch.py --dataset svhn --num_labeled 250 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120
python scripts/fixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120
python scripts/fixmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120
python scripts/fixmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120
python scripts/flexmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120
python scripts/flexmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120
python scripts/flexmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120
python scripts/mixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120
python scripts/mixmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120
python scripts/mixmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/fixmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/fixmatch.py --dataset cifar10 --num_labeled 4000 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/fixmatch.py --dataset cifar10 --num_labeled 4000 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/flexmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/flexmatch.py --dataset cifar10 --num_labeled 4000 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/flexmatch.py --dataset cifar10 --num_labeled 4000 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/mixmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/mixmatch.py --dataset cifar10 --num_labeled 4000 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/mixmatch.py --dataset cifar10 --num_labeled 4000 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.9 --max_minutes 120
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 4000 --seed 308 --target_acc 0.9 --max_minutes 120
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 4000 --seed 2701 --target_acc 0.9 --max_minutes 120
python scripts/efficientmatch.py --dataset cifar100 --num_labeled 2500 --seed 2312 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/efficientmatch.py --dataset cifar100 --num_labeled 2500 --seed 308 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/efficientmatch.py --dataset cifar100 --num_labeled 2500 --seed 2701 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/fixmatch.py --dataset cifar100 --num_labeled 2500 --seed 2312 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/fixmatch.py --dataset cifar100 --num_labeled 2500 --seed 308 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/fixmatch.py --dataset cifar100 --num_labeled 2500 --seed 2701 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/flexmatch.py --dataset cifar100 --num_labeled 2500 --seed 2312 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/flexmatch.py --dataset cifar100 --num_labeled 2500 --seed 308 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/flexmatch.py --dataset cifar100 --num_labeled 2500 --seed 2701 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/mixmatch.py --dataset cifar100 --num_labeled 2500 --seed 2312 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/mixmatch.py --dataset cifar100 --num_labeled 2500 --seed 308 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/mixmatch.py --dataset cifar100 --num_labeled 2500 --seed 2701 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/regmixmatch.py --dataset cifar100 --num_labeled 2500 --seed 2312 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/regmixmatch.py --dataset cifar100 --num_labeled 2500 --seed 308 --target_acc 0.5 --max_minutes 120 --widen_factor 4
python scripts/regmixmatch.py --dataset cifar100 --num_labeled 2500 --seed 2701 --target_acc 0.5 --max_minutes 120 --widen_factor 4

# freematch -- Table 4
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120 --freematch_threshold True
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 308 --target_acc 0.9 --max_minutes 120 --freematch_threshold True
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 2701 --target_acc 0.9 --max_minutes 120 --freematch_threshold True
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --freematch_threshold True
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --freematch_threshold True
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --freematch_threshold True
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.9 --max_minutes 120 --freematch_threshold True
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 308 --target_acc 0.9 --max_minutes 120 --freematch_threshold True
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2701 --target_acc 0.9 --max_minutes 120 --freematch_threshold True
python scripts/efficientmatch.py --dataset cifar100 --num_labeled 2500 --seed 2312 --target_acc 0.5 --max_minutes 120 --widen_factor 4 --freematch_threshold True
python scripts/efficientmatch.py --dataset cifar100 --num_labeled 2500 --seed 308 --target_acc 0.5 --max_minutes 120 --widen_factor 4 --freematch_threshold True
python scripts/efficientmatch.py --dataset cifar100 --num_labeled 2500 --seed 2701 --target_acc 0.5 --max_minutes 120 --widen_factor 4 --freematch_threshold True

# freematch-noclamp -- Appendix A.5
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120 --freematch_threshold True --freematch_svhn_clamp False
python scripts/regmixmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120 --svhn_clamp False

# mu -- Tables 5, 6
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --mu 1
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --mu 1
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --mu 1
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --mu 5
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --mu 5
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --mu 5
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --mu 7
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --mu 7
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --mu 7
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.9 --max_minutes 120 --mu 1
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 308 --target_acc 0.9 --max_minutes 120 --mu 1
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2701 --target_acc 0.9 --max_minutes 120 --mu 1
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.9 --max_minutes 120 --mu 5
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 308 --target_acc 0.9 --max_minutes 120 --mu 5
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2701 --target_acc 0.9 --max_minutes 120 --mu 5
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.9 --max_minutes 120 --mu 7
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 308 --target_acc 0.9 --max_minutes 120 --mu 7
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 4000 --seed 2701 --target_acc 0.9 --max_minutes 120 --mu 7

# lambda-mix -- Table 7
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --mixup_weight 0.5
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --mixup_weight 0.5
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --mixup_weight 0.5
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --mixup_weight 2.0
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --mixup_weight 2.0
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --mixup_weight 2.0

# mixing-target -- Table 8
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --mixing_target hard
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --mixing_target hard
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --mixing_target hard
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120 --mixing_target hard
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 308 --target_acc 0.9 --max_minutes 120 --mixing_target hard
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 2701 --target_acc 0.9 --max_minutes 120 --mixing_target hard
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --mixing_target soft
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --mixing_target soft
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --mixing_target soft
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.9 --max_minutes 120 --mixing_target soft
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 308 --target_acc 0.9 --max_minutes 120 --mixing_target soft
python scripts/efficientmatch.py --dataset svhn --num_labeled 250 --seed 2701 --target_acc 0.9 --max_minutes 120 --mixing_target soft

# matched-mu -- Table 10
python scripts/fixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --mu 3
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.8 --max_minutes 120 --mu 3 --static_shapes True
python scripts/fixmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --mu 3
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 250 --seed 308 --target_acc 0.8 --max_minutes 120 --mu 3 --static_shapes True
python scripts/fixmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --mu 3
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 250 --seed 2701 --target_acc 0.8 --max_minutes 120 --mu 3 --static_shapes True

# asymptotic -- Table 11
python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed 42 --tag unlimited
python scripts/regmixmatch.py --dataset cifar10 --num_labeled 250 --seed 42 --tag unlimited
```

## Two runs that ignore the budget

The last two, under `asymptotic`, deliberately carry no target and no time cap: they run the full
2^20-iteration horizon to see where each method plateaus, and take over twelve hours each. They are
also the only ones on seed 42. See docs/asymptotic.md.
