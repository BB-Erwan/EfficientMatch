#!/bin/bash
# Polling-based watchdog (tail -f | grep -m1 does not reliably terminate the
# pipeline on match under Git Bash on Windows -- grep exits but tail keeps
# blocking, so the wait never returns without an external timeout wrapped
# around it. Polling with grep -c avoids relying on that broken signaling).
LOG="C:\Users\bertranh\AppData\Local\Temp\claude\c--Users-bertranh-Documents-GitHub-EfficientMatch\c6400e03-5c9f-42c6-ad6c-470c0b5b5d56\tasks\bzeyqanxd.output"
TIMEOUT_S=7200
POLL_S=30

for seed in 2312 0308 2701; do
  for method in fixmatch flexmatch mixmatch; do
    marker="${method}.py --dataset cifar100 --widen_factor 8 --num_labeled 10000 --seed ${seed} --target_acc 0.60"

    # Wait for this run to start
    while ! grep -qF "$marker" "$LOG" 2>/dev/null; do
      sleep "$POLL_S"
    done
    start_ts=$(date +%s)
    echo "WATCH_START ${method} seed=${seed} at $(date)"

    # Count how many "Reached target_acc" lines already existed before this run started,
    # so we can detect a NEW one (this run's own completion) rather than an earlier run's.
    baseline=$(grep -cF "Reached target_acc" "$LOG" 2>/dev/null || echo 0)

    while true; do
      sleep "$POLL_S"
      now_ts=$(date +%s)
      elapsed=$((now_ts - start_ts))
      current=$(grep -cF "Reached target_acc" "$LOG" 2>/dev/null || echo 0)
      if [ "$current" -gt "$baseline" ]; then
        echo "DONE ${method} seed=${seed} in ${elapsed}s (within 2h)"
        break
      fi
      if [ "$elapsed" -ge "$TIMEOUT_S" ]; then
        echo "TIMEOUT ${method} seed=${seed} after ${elapsed}s (>${TIMEOUT_S}s) — killing"
        pid=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { \$_.CommandLine -match '${method}.py' -and \$_.CommandLine -match 'seed ${seed}' -and \$_.CommandLine -match 'num_labeled 10000' }).ProcessId" | tr -d '\r')
        if [ -n "$pid" ]; then
          powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force" > /dev/null 2>&1
          echo "KILLED ${method} seed=${seed} PID=$pid after ${elapsed}s"
        else
          echo "WARNING: could not find PID to kill for ${method} seed=${seed}"
        fi
        break
      fi
    done
  done
done
