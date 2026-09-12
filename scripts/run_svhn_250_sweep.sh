#!/bin/bash
set -x
cd "C:/Users/bertranh/Documents/GitHub/EfficientMatch/scripts"
for seed in 2312 0308 2701; do
  for method in fixmatch mixmatch flexmatch efficientmatch; do
    conda run -n DEEP-GPU python ${method}.py --dataset svhn --num_labeled 250 --seed ${seed} --target_acc 0.90
  done
done
