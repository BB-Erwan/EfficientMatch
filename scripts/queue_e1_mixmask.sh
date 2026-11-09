#!/usr/bin/env bash
# Ablation E1: which confidence mask gates L_mix (CIFAR-10/250, target 80%, mu = 3).
# The reference 'partner' runs already exist (efficientmatch_ema_metrics.json); this queue runs the
# four other variants, seed-first, one run at a time, and never re-runs an existing result.
# Each run stops at the target or at the protocol's 2-hour cap (--max_minutes 120); `timeout`
# is a backup watchdog at 130 minutes. Logs: logs/e1_mixmask/.
#
#   bash scripts/queue_e1_mixmask.sh
set -u
cd "$(dirname "$0")/.."
source /c/Users/bertranh/anaconda3/etc/profile.d/conda.sh
conda activate DEEP-GPU
mkdir -p logs/e1_mixmask

for seed in 2312 308 2701; do
  # premix and both first: they implement the original design intent (only labeled samples and
  # admitted pseudo-labels supervise the mix).
  for mask in premix both none anchor; do
    out="results/cifar10-labeled-250-seed-${seed}/efficientmatch_ema_mixmask_${mask}_metrics.json"
    log="logs/e1_mixmask/seed${seed}_${mask}.log"
    if [ -f "$out" ]; then
      echo "$(date '+%F %T') skip (exists) seed=${seed} mask=${mask}"
      continue
    fi
    echo "$(date '+%F %T') start seed=${seed} mask=${mask}"
    timeout 130m python scripts/efficientmatch.py --dataset cifar10 --num_labeled 250 --seed "$seed" \
      --target_acc 0.80 --max_minutes 120 --mix_mask "$mask" > "$log" 2>&1
    echo "$(date '+%F %T') end seed=${seed} mask=${mask} exit=$?"
  done
done
echo "$(date '+%F %T') queue done"
