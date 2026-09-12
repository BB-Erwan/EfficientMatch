#!/bin/bash
LOG="C:\Users\bertranh\AppData\Local\Temp\claude\c--Users-bertranh-Documents-GitHub-EfficientMatch\c6400e03-5c9f-42c6-ad6c-470c0b5b5d56\tasks\bv4bel9xo.output"

tail -n +1 -f "$LOG" | grep -m 1 --line-buffered "flexmatch.py --dataset svhn --num_labeled 250 --seed 2312" > /dev/null
echo "Started watching flexmatch seed 2312 evaluations"

count=0
all_low=1
tail -n +1 -f "$LOG" | grep --line-buffered -oE "Acc: [0-9.]+" | while read -r line; do
    acc=$(echo "$line" | sed 's/Acc: //')
    count=$((count+1))
    is_low=$(awk -v a="$acc" 'BEGIN{print (a<0.10)?1:0}')
    if [ "$is_low" -eq 0 ]; then
        all_low=0
    fi
    echo "flexmatch eval $count/5: acc=$acc"
    if [ "$count" -ge 5 ]; then
        if [ "$all_low" -eq 1 ]; then
            echo "FLEXMATCH_STUCK: 5 evaluations all under 10% accuracy — killing flexmatch"
            pid=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { \$_.CommandLine -match 'flexmatch.py' -and \$_.CommandLine -match 'seed 2312' }).ProcessId" | tr -d '\r')
            if [ -n "$pid" ]; then
                powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force" > /dev/null 2>&1
                echo "Killed flexmatch PID $pid"
            fi
        else
            echo "FLEXMATCH_OK: not all 5 evaluations under 10% — leaving flexmatch running"
        fi
        break
    fi
done
