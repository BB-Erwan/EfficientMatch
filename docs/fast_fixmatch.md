# Fast FixMatch: FLOP gain against wall-clock overhead

Backs **Table 9** and **Appendix A.4** of the paper, and explains why Fast FixMatch
(Chen et al., 2024) — the closest work in objective — is not part of the main comparison.

```bash
python scripts/benchmark_fast_fixmatch.py
```

## Why it is not in Tables 2 and 3

Fast FixMatch's curriculum batch size (CBS) explicitly varies the unlabeled batch size at every
iteration, following a B-EXP schedule:

```
B(t) = u · (1 − (1 − t/T) / ((1 − alpha) + alpha · (1 − t/T)))      alpha = 0.7
```

so the tensor shapes change from one iteration to the next. Every other method in this repository
trains on fixed shapes. Comparing convergence budgets across that boundary would compare two
different execution regimes rather than two algorithms, so the method is measured separately, on a
targeted diagnostic instead of a full convergence run.

## The diagnostic

CIFAR-10, 250 labels, seed 2312, **3000 iterations for both methods**, on the standard pipeline used
for every other result. 3000 iterations is enough for the CBS curve to sweep its whole range once
(batch 8 up to 448), which is what the measurement is about.

| Method | Wall-clock | Accumulated GFLOPs | ms / iteration |
|---|---:|---:|---:|
| FixMatch, fixed batch of 448 | 197.9 s | 2 550 300 | 65.96 |
| Fast FixMatch, CBS from 8 to 448 | 421.0 s | 953 154 | 140.32 |
| **Ratio (Fast FixMatch / FixMatch)** | **2.13x** | **0.37x** | **2.13x** |

**The FLOP gain is confirmed.** Fast FixMatch uses 62.6% fewer FLOPs than FixMatch over the same
3000 iterations, consistent with the gain its authors report.

**The wall-clock overhead neutralises and reverses it.** Fast FixMatch completes those 3000
iterations more than twice as slowly in practice, despite a strictly lower compute budget. The
execution log attributes the overhead directly to CUDA Graphs recompilation: `torch.compile` in
`reduce-overhead` mode records a new graph for every distinct batch shape it encounters — nine
distinct sizes over the 8 to 448 range — rather than reusing an already-compiled one, a mechanism
fundamentally incompatible with variable tensor shapes.

This is a limitation of the compilation pipeline used here, not a challenge to the authors' original
result on the FLOP metric, which this work itself retains as one of its three criteria. It is also a
concrete illustration of why the paper reports all three: a method can reduce compute and still cost
more time, and neither metric alone would show it.
