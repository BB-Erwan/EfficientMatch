#!/usr/bin/env bash
# Follow-up to queue_rebuttal_gpu2.sh: the full Fast FixMatch run of seed 2312 crashed on a Windows
# file lock (fixed in common/recording.py); its re-run was dropped after seed 308 stayed at 69.4% after the 2-hour
# budget (it does not reach the target). When the main queue reaches its last step (the 16-hour
# FixMatch run), stop that run right away, run MixMatch with the
# published L_x (--lx_target mixed) on CIFAR-10/250 and CIFAR-10/4000, seed 2312, run EfficientMatch
# without the mask (--mix_mask none) on SVHN/250, CIFAR-10/4000 and CIFAR-100/2500 (3 seeds, seed-first),
# run MixMatch and FlexMatch for a full 2-hour budget without target on CIFAR-10/250 seed 42 (threshold-free
# comparison, with the existing full-horizon EfficientMatch/RegMixMatch runs), then restart FixMatch.
set -u
cd "$(dirname "$0")/.."
source /c/Users/bertranh/anaconda3/etc/profile.d/conda.sh
conda activate DEEP-GPU
until [ "$(grep -c ' start ' logs/gpu2_queue.log)" -ge 7 ] || grep -q 'queue done' logs/gpu2_queue.log; do sleep 20; done
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'fixmatch.py --dataset cifar10 --num_labeled 250 --seed 42' -and \$_.CommandLine -notmatch 'fast_fixmatch' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
sleep 5
partial="results/cifar10-labeled-250-seed-42/fixmatch_ema_unlimited_metrics.json"
[ -f "$partial" ] && mv "$partial" "results/cifar10-labeled-250-seed-42/fixmatch_ema_unlimited_killed_metrics.json"
for nl in 250 4000; do
  if [ "$nl" = 250 ]; then tgt=0.80; else tgt=0.90; fi
  out="results/cifar10-labeled-${nl}-seed-2312/mixmatch_ema_lxmixed_metrics.json"
  if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) mixmatch_lxmixed_${nl}"; continue; fi
  echo "$(date '+%F %T') start mixmatch_lxmixed_${nl}_seed2312"
  timeout 130m python scripts/mixmatch.py --dataset cifar10 --num_labeled "$nl" --seed 2312     --target_acc "$tgt" --max_minutes 120 --lx_target mixed > "logs/gpu2/mixmatch_lxmixed_${nl}_seed2312.log" 2>&1
  rc=$?; echo "$(date '+%F %T') end mixmatch_lxmixed_${nl}_seed2312 exit=$rc"
done
for seed in 2312 308 2701; do
  for cfg in "svhn 250 0.90 2" "cifar10 4000 0.90 2" "cifar100 2500 0.50 4"; do
    set -- $cfg; ds=$1; nl=$2; tgt=$3; wf=$4
    suffix=""; [ "$wf" != 2 ] && suffix="_wf${wf}"
    out="results/${ds}-labeled-${nl}-seed-${seed}/efficientmatch_ema_mixmask_none${suffix}_metrics.json"
    name="none_${ds}_${nl}_seed${seed}"
    if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) $name"; continue; fi
    echo "$(date '+%F %T') start $name"
    timeout 130m python scripts/efficientmatch.py --dataset "$ds" --num_labeled "$nl" --seed "$seed"       --widen_factor "$wf" --target_acc "$tgt" --max_minutes 120 --mix_mask none > "logs/gpu2/${name}.log" 2>&1
    rc=$?; echo "$(date '+%F %T') end $name exit=$rc"
  done
done
for m in mixmatch flexmatch; do
  out="results/cifar10-labeled-250-seed-42/${m}_ema_budget2h_metrics.json"
  name="${m}_budget2h_seed42"
  if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) $name"; continue; fi
  echo "$(date '+%F %T') start $name"
  timeout 130m python "scripts/${m}.py" --dataset cifar10 --num_labeled 250 --seed 42 --max_minutes 120     --tag budget2h > "logs/gpu2/${name}.log" 2>&1
  rc=$?; echo "$(date '+%F %T') end $name exit=$rc"
done
echo "$(date '+%F %T') start fixmatch_full_seed42"
timeout 20h python scripts/fixmatch.py --dataset cifar10 --num_labeled 250 --seed 42 --tag unlimited \
  > logs/gpu2/fixmatch_full_seed42.log 2>&1
rc=$?; echo "$(date '+%F %T') end fixmatch_full_seed42 exit=$rc"
echo "$(date '+%F %T') follow-up done"
