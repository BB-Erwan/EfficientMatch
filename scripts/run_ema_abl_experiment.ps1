# Run fixmatch, flexmatch and mixmatch on CIFAR-10 (250 labels) until they reach 85% test accuracy.
# max_steps is a safety net (2^20) in case a method never reaches the target.
#
# Usage:
#   & "C:\Users\erwan.bertrand\AppData\Local\anaconda3\shell\condabin\conda-hook.ps1" ; conda activate DEEP-GPU
#   .\scripts\run_target_acc_experiment.ps1

python scripts\efficientmatch.py --target_acc 0.85 --max_steps 1048576 --seed 666
python scripts\efficientmatch.py --target_acc 0.85 --max_steps 1048576 --seed 666 --use_ema true

