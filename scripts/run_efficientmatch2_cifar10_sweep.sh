#!/bin/bash
set -x
cd "C:/Users/bertranh/Documents/GitHub/EfficientMatch/scripts"
for seed in 2312 0308 2701; do
  conda run -n DEEP-GPU python efficientmatch_2.py --dataset cifar10 --num_labeled 250 --seed ${seed} --target_acc 0.80 --adaptive_threshold false
done
