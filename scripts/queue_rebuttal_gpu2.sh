#!/usr/bin/env bash
# GPU queue, in order, one run at a time, never re-running an existing result:
#   1. Full Fast FixMatch (CBS + strong labeled augmentation + FlexMatch CPL, CBS horizon = 2^20 as
#      in Chen et al.), eager mode, CIFAR-10/250, target 80%, 3 seeds.
#   2. Mixup-channel control: EfficientMatch with the mix replaced by an extra strong view
#      (--mix_input extra_strong --mix_mask none), CIFAR-10/250, target 80%, 3 seeds.
#   3. FixMatch for the full 2^20-iteration schedule, CIFAR-10/250, seed 42 (no target, no cap),
#      to compare with the existing full EfficientMatch run on the same seed.
#
#   bash scripts/queue_rebuttal_gpu2.sh
set -u
cd "$(dirname "$0")/.."
source /c/Users/bertranh/anaconda3/etc/profile.d/conda.sh
conda activate DEEP-GPU
mkdir -p logs/gpu2

run() {  # run <result file> <log name> <timeout> <command...>
  local out="$1" log="logs/gpu2/$2.log" to="$3"; shift 3
  if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) $2"; return; fi
  echo "$(date '+%F %T') start $2"
  timeout "$to" "$@" > "$log" 2>&1
  echo "$(date '+%F %T') end $2 exit=$?"
}

for seed in 2312 308 2701; do
  run "results/cifar10-labeled-250-seed-${seed}/fast_fixmatch_ema_full_cbs1048576_eager_metrics.json" \
      "ffm_full_seed${seed}" 130m \
      python scripts/fast_fixmatch.py --dataset cifar10 --num_labeled 250 --seed "$seed" \
        --target_acc 0.80 --max_minutes 120 --full true --cbs_total_steps 1048576 --compile false
done
for seed in 2312 308 2701; do
  run "results/cifar10-labeled-250-seed-${seed}/efficientmatch_ema_mixmask_none_extrastrong_metrics.json" \
      "extrastrong_seed${seed}" 130m \
      python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed "$seed" \
        --target_acc 0.80 --max_minutes 120 --mix_input extra_strong --mix_mask none
done
run "results/cifar10-labeled-250-seed-42/fixmatch_ema_unlimited_metrics.json" "fixmatch_full_seed42" 20h \
    python scripts/fixmatch.py --dataset cifar10 --num_labeled 250 --seed 42 --tag unlimited
echo "$(date '+%F %T') queue done"
