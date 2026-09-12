import json
import os

d = r"C:\Users\bertranh\Documents\GitHub\EfficientMatch\results\cifar10-labeled-250-seed-2312"
for fn in ["fixmatch_ema_metrics.json", "mixmatch_ema_metrics.json", "flexmatch_ema_metrics.json", "efficientmatch_ema_metrics.json"]:
    p = os.path.join(d, fn)
    with open(p) as f:
        data = json.load(f)
    t = data.get("time_elapsed", [])
    acc = data.get("test_acc", [])
    print(fn)
    for i in range(min(5, len(t))):
        print(f"  eval {i+1}: t={t[i]:.1f}s acc={acc[i]:.4f}")
    # time and acc at ~14 min = 840s
    for i, tv in enumerate(t):
        if tv >= 840:
            print(f"  at t>=840s (14min): eval {i+1}, t={tv:.1f}s, acc={acc[i]:.4f}")
            break
