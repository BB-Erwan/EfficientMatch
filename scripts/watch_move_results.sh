#!/bin/bash
LOG="C:\Users\bertranh\AppData\Local\Temp\claude\c--Users-bertranh-Documents-GitHub-EfficientMatch\c6400e03-5c9f-42c6-ad6c-470c0b5b5d56\tasks\b0ff2g830.output"
RESULTS="C:/Users/bertranh/Documents/GitHub/EfficientMatch/results"

tail -n +1 -f "$LOG" | grep -m 1 --line-buffered "efficientmatch_2.py --dataset cifar10 --num_labeled 250 --seed 0308 --target_acc 0.80" > /dev/null
echo "Started watching efficientmatch_2 seed 0308 for completion"
tail -n +1 -f "$LOG" | grep -m 1 --line-buffered "Reached target_acc=0.8000" > /dev/null
sleep 2
src="$RESULTS/labeled-250-seed-308/efficientmatch_2_ema_metrics.json"
dst_dir="$RESULTS/cifar10-labeled-250-seed-308"
mkdir -p "$dst_dir"
if [ -f "$src" ]; then
    mv "$src" "$dst_dir/efficientmatch_2_ema_metrics.json"
    rmdir "$RESULTS/labeled-250-seed-308" 2>/dev/null
    echo "MOVED seed 0308 result to $dst_dir/efficientmatch_2_ema_metrics.json"
else
    echo "WARNING: source file not found for seed 0308 at $src"
fi
