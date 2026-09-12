#!/bin/bash
LOG="C:\Users\bertranh\AppData\Local\Temp\claude\c--Users-bertranh-Documents-GitHub-EfficientMatch\c6400e03-5c9f-42c6-ad6c-470c0b5b5d56\tasks\bv4bel9xo.output"
SEEDS="0308 2701"

for seed in $SEEDS; do
    tail -n +1 -f "$LOG" | grep -m 1 --line-buffered "efficientmatch.py --dataset svhn --num_labeled 250 --seed $seed" > /dev/null
    echo "SKIP: efficientmatch seed $seed started — killing immediately"
    sleep 2
    pid=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { \$_.CommandLine -match 'efficientmatch.py' -and \$_.CommandLine -match \"seed $seed\" }).ProcessId" | tr -d '\r')
    if [ -n "$pid" ]; then
        powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force" > /dev/null 2>&1
        echo "Killed efficientmatch seed $seed PID $pid"
    fi
done
