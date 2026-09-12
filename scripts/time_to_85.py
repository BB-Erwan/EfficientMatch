import json
import glob
import os

TEST_PERIOD = 500
THRESH = 0.85

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(repo_root, "results", "svhn-labeled-250-seed-*", "*_metrics.json")))
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    acc = data.get("test_acc", [])
    t = data.get("time_elapsed", [])
    if not acc:
        print(f"{f}: EMPTY")
        continue
    hit = None
    for i, a in enumerate(acc):
        if a >= THRESH:
            hit = i
            break
    if hit is None:
        print(f"{f}: never reached {THRESH*100:.0f}% (max={max(acc)*100:.2f}%)")
    else:
        step = TEST_PERIOD * hit if hit >= 1 else 1
        print(f"{f}: reached {THRESH*100:.0f}% at eval #{hit+1} (~step {step}), time={t[hit]:.1f}s ({t[hit]/60:.1f} min), acc={acc[hit]:.4f}")
