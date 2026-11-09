#!/usr/bin/env bash
# Replaces the rest of queue_gpu2_followup.sh with a new order: EfficientMatch without the mask
# (--mix_mask none) on SVHN/250, CIFAR-10/4000 and CIFAR-100/2500 (3 seeds, seed-first) comes first,
# because it may change why EfficientMatch works; then MixMatch with the published L_x on CIFAR-10/4000,
# the 2-hour MixMatch/FlexMatch runs without target (CIFAR-10/250 seed 42), and FixMatch for 2^20
# iterations (seed 42). Waits for the MixMatch --lx_target mixed run already on the GPU to end.
set -u
cd "$(dirname "$0")/.."
source /c/Users/bertranh/anaconda3/etc/profile.d/conda.sh
conda activate DEEP-GPU
gpu_busy() {
  powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'scripts/(mixmatch|fixmatch|flexmatch|efficientmatch|regmixmatch|fast_fixmatch)\.py' }).Count" | tr -d '\r'
}
while [ "$(gpu_busy)" != "0" ]; do sleep 30; done
echo "$(date '+%F %T') GPU free, starting"
for seed in 2312 308 2701; do
  for cfg in "svhn 250 0.90 2" "cifar10 4000 0.90 2" "cifar100 2500 0.50 4"; do
    set -- $cfg; ds=$1; nl=$2; tgt=$3; wf=$4
    suffix=""; [ "$wf" != 2 ] && suffix="_wf${wf}"
    out="results/${ds}-labeled-${nl}-seed-${seed}/efficientmatch_ema_mixmask_none${suffix}_metrics.json"
    name="none_${ds}_${nl}_seed${seed}"
    if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) $name"; continue; fi
    echo "$(date '+%F %T') start $name"
    timeout 130m python scripts/efficientmatch.py --dataset "$ds" --num_labeled "$nl" --seed "$seed" \
      --widen_factor "$wf" --target_acc "$tgt" --max_minutes 120 --mix_mask none > "logs/gpu2/${name}.log" 2>&1
    rc=$?; echo "$(date '+%F %T') end $name exit=$rc"
  done
done
out="results/cifar10-labeled-4000-seed-2312/mixmatch_ema_lxmixed_metrics.json"
if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) mixmatch_lxmixed_4000"; else
  echo "$(date '+%F %T') start mixmatch_lxmixed_4000_seed2312"
  timeout 130m python scripts/mixmatch.py --dataset cifar10 --num_labeled 4000 --seed 2312 \
    --target_acc 0.90 --max_minutes 120 --lx_target mixed > logs/gpu2/mixmatch_lxmixed_4000_seed2312.log 2>&1
  rc=$?; echo "$(date '+%F %T') end mixmatch_lxmixed_4000_seed2312 exit=$rc"
fi
for m in mixmatch flexmatch; do
  out="results/cifar10-labeled-250-seed-42/${m}_ema_budget2h_metrics.json"
  name="${m}_budget2h_seed42"
  if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) $name"; continue; fi
  echo "$(date '+%F %T') start $name"
  timeout 130m python "scripts/${m}.py" --dataset cifar10 --num_labeled 250 --seed 42 --max_minutes 120 \
    --tag budget2h > "logs/gpu2/${name}.log" 2>&1
  rc=$?; echo "$(date '+%F %T') end $name exit=$rc"
done
out="results/cifar10-labeled-250-seed-42/fixmatch_ema_unlimited_metrics.json"
if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) fixmatch_full_seed42"; else
  echo "$(date '+%F %T') start fixmatch_full_seed42"
  timeout 20h python scripts/fixmatch.py --dataset cifar10 --num_labeled 250 --seed 42 --tag unlimited \
    > logs/gpu2/fixmatch_full_seed42.log 2>&1
  rc=$?; echo "$(date '+%F %T') end fixmatch_full_seed42 exit=$rc"
fi
echo "$(date '+%F %T') follow-up 2 done"
