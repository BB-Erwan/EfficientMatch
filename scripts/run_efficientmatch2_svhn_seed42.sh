#!/bin/bash
set -x
cd "C:/Users/bertranh/Documents/GitHub/EfficientMatch/scripts"
# Wait for the seed-308 completion sweep (fixmatch/flexmatch/mixmatch) to fully finish.
# Require two consecutive clean checks to avoid a race where the previous
# process has logged completion but not yet exited.
clean_checks=0
while [ "$clean_checks" -lt 2 ]; do
  if pgrep -f "run_seed308_complete.sh" > /dev/null || pgrep -f "(fixmatch|flexmatch|mixmatch)\.py --dataset cifar10 --num_labeled 250 --seed 0308" > /dev/null; then
    clean_checks=0
  else
    clean_checks=$((clean_checks+1))
  fi
  sleep 5
done
conda run -n DEEP-GPU python efficientmatch_2.py --dataset svhn --num_labeled 250 --seed 42 --target_acc 0.90 --adaptive_threshold false
