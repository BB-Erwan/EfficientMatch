import json
import glob

files = sorted(glob.glob("results/svhn-labeled-250-seed-*/*_metrics.json"))
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    acc = data.get("test_acc", [])
    if acc:
        print(f"{f}: n_evals={len(acc)} last={acc[-1]:.4f} max={max(acc):.4f}")
    else:
        print(f"{f}: EMPTY")
