# Run fixmatch, flexmatch and mixmatch (EMA) on CIFAR-10 (250 labels) until they reach 80% test accuracy, seed 2312.
python scripts\fixmatch.py --target_acc 0.80 --max_steps 1048576 --seed 2312 --use_ema true
python scripts\flexmatch.py --target_acc 0.80 --max_steps 1048576 --seed 2312 --use_ema true
python scripts\mixmatch.py --target_acc 0.80 --max_steps 1048576 --seed 2312 --use_ema true
