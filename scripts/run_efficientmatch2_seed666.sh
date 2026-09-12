#!/bin/bash
set -x
cd "C:/Users/bertranh/Documents/GitHub/EfficientMatch/scripts"
# Wait for the current sweep (seed 2701) to finish before starting
while pgrep -f "efficientmatch_2.py.*seed 2701" > /dev/null; do
  sleep 5
done
conda run -n DEEP-GPU python efficientmatch_2.py --dataset cifar10 --num_labeled 250 --seed 666 --target_acc 0.80 --adaptive_threshold false
