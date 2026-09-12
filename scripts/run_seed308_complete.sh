#!/bin/bash
set -x
cd "C:/Users/bertranh/Documents/GitHub/EfficientMatch/scripts"
for method in fixmatch flexmatch mixmatch; do
  conda run -n DEEP-GPU python ${method}.py --dataset cifar10 --num_labeled 250 --seed 0308 --target_acc 0.80
done
