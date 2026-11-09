"""AUC restricted to the untruncated window: for each config and seed, H = the earliest point at
which any method stops at its target (along corrected time or FLOPs). Within [0, H] no curve is
held, so the score does not depend on the targets."""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['x']
import run_analysis as ra
from types import SimpleNamespace

CFGS = [('svhn', 250, 0.90), ('cifar10', 250, 0.80), ('cifar10', 4000, 0.90), ('cifar100', 2500, 0.50)]
for axis in ['time', 'flops']:
    print(f'\n===== axis: {axis} =====')
    for ds, nl, tgt in CFGS:
        methods = [m + '_wf4' for m in ra.MAIN_METHODS] if ds == 'cifar100' else ra.MAIN_METHODS
        ranks_all = {m: [] for m in methods}
        vals_all = {m: [] for m in methods}
        hs = []
        for seed in [2312, 308, 2701]:
            args = SimpleNamespace(dataset=ds, num_labeled=nl, seed=seed, dir=None, default_widen_factor=None)
            rdir = ra.resolve_dir(args)
            runs = ra.load_runs(rdir, ra.default_widen_factor(args, rdir))
            curves, stops = {}, []
            for m in methods:
                data, wf = runs[m]
                xs, accs = ra.held_curve(data, wf, m, ds, axis, 120)
                curves[m] = (xs, accs)
                hit = [x for x, a in zip(xs, accs) if a >= tgt]
                if hit:
                    stops.append(hit[0])
            H = min(stops)
            hs.append(H)
            v = {m: ra.mean_acc_over(*curves[m], H) * 100 for m in methods}
            order = sorted(v, key=lambda m: -v[m])
            for r, m in enumerate(order):
                ranks_all[m].append(r + 1)
                vals_all[m].append(v[m])
        print(f'{ds}/{nl}: H per seed = ' + ', '.join(f'{h:.1f}' for h in hs) + (' min' if axis == 'time' else ' PFLOPs'))
        for m in methods:
            print(f'   {m.split("_")[0]:<15} mean acc {sum(vals_all[m])/3:6.2f}   ranks {ranks_all[m]}')
