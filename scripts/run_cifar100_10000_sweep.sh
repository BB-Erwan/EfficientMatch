#!/bin/bash
set -x
cd "C:/Users/bertranh/Documents/GitHub/EfficientMatch/scripts"
for seed in 2312 0308 2701; do
  for method in fixmatch flexmatch mixmatch; do
    conda run -n DEEP-GPU python ${method}.py --dataset cifar100 --widen_factor 8 --num_labeled 10000 --seed ${seed} --target_acc 0.60 --test_period 256
  done
done
