#!/usr/bin/env bash
# Follow-up 4: MixMatch with the published L_x (--lx_target mixed) on CIFAR-10/4000, seeds 308 and 2701
# (seed 2312 is done). Waits for the FixMatch 2^20 run of follow-up 3 (last job of that queue) to end.
set -u
cd "$(dirname "$0")/.."
source /c/Users/bertranh/anaconda3/etc/profile.d/conda.sh
conda activate DEEP-GPU
gpu_busy() {
  powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'scripts/(mixmatch|fixmatch|flexmatch|efficientmatch|regmixmatch|fast_fixmatch)\.py' }).Count" | tr -d '\r'
}
while [ "$(gpu_busy)" != "0" ]; do sleep 30; done
echo "$(date '+%F %T') GPU free, starting"
for seed in 308 2701; do
  out="results/cifar10-labeled-4000-seed-${seed}/mixmatch_ema_lxmixed_metrics.json"
  name="mixmatch_lxmixed_4000_seed${seed}"
  if [ -f "$out" ]; then echo "$(date '+%F %T') skip (exists) $name"; continue; fi
  echo "$(date '+%F %T') start $name"
  timeout 130m python scripts/mixmatch.py --dataset cifar10 --num_labeled 4000 --seed "$seed" \
    --target_acc 0.90 --max_minutes 120 --lx_target mixed > "logs/gpu2/${name}.log" 2>&1
  rc=$?; echo "$(date '+%F %T') end $name exit=$rc"
done
echo "$(date '+%F %T') follow-up 4 done"
