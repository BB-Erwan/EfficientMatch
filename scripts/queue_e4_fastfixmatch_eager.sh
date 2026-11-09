#!/usr/bin/env bash
# Fast FixMatch (curriculum batch size only) in eager mode, CIFAR-10/250, target 80%, 3 seeds.
# Eager mode avoids the CUDA-graph re-recording that variable batch shapes trigger under
# torch.compile (docs/fast_fixmatch.md). Each run stops at the target or at the 2-hour cap;
# `timeout` is a backup watchdog. Never re-runs an existing result.
#
#   bash scripts/queue_e4_fastfixmatch_eager.sh
set -u
cd "$(dirname "$0")/.."
source /c/Users/bertranh/anaconda3/etc/profile.d/conda.sh
conda activate DEEP-GPU
mkdir -p logs/e4_fastfixmatch
for seed in 2312 308 2701; do
  out="results/cifar10-labeled-250-seed-${seed}/fast_fixmatch_ema_eager_metrics.json"
  log="logs/e4_fastfixmatch/seed${seed}.log"
  if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) seed=${seed}"; continue; fi
  echo "$(date '+%F %T') start seed=${seed}"
  timeout 130m python scripts/fast_fixmatch.py --dataset cifar10 --num_labeled 250 --seed "$seed" \
    --target_acc 0.80 --max_minutes 120 --compile false > "$log" 2>&1
  echo "$(date '+%F %T') end seed=${seed} exit=$?"
done
echo "$(date '+%F %T') queue done"
