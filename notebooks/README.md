# Notebooks

The same code as `scripts/`, laid out to be read and modified cell by cell.

<p align="center">
  <img src="../figures/augmentation_views.png" width="85%" alt="Weak and strong views of eight unlabeled images">
</p>

*Section 3 of every method notebook draws this: eight unlabeled CIFAR-10 images, their weak view
(flip and crop) and their strong view (RandAugment, then flip and crop). The eight images form one
batch, so they share one crop and one set of RandAugment operations, as in every run of the paper.*

| Notebook | What it contains |
|---|---|
| [efficientmatch.ipynb](efficientmatch.ipynb) | The method of the paper: FixMatch's losses plus the confidence-filtered Mixup channel |
| [fixmatch.ipynb](fixmatch.ipynb) | FixMatch, the consistency baseline |
| [flexmatch.ipynb](flexmatch.ipynb) | FlexMatch, FixMatch with a per-class adaptive threshold |
| [mixmatch.ipynb](mixmatch.ipynb) | MixMatch, Mixup without confidence filtering |
| [regmixmatch.ipynb](regmixmatch.ipynb) | RegMixMatch, FreeMatch's adaptive threshold plus a confidence-routed ResizeMix |
| [figures.ipynb](figures.ipynb) | Figures 1 to 5 of the paper, and a section to plot your own runs |

## Running them

```bash
pip install -r requirements.txt jupyter
jupyter lab notebooks/
```

VS Code, or any other Jupyter front end, works as well. A CUDA GPU is expected for the method
notebooks; `figures.ipynb` only reads the files in `results/` and runs anywhere in a few seconds.

## How the method notebooks are organised

All five follow the same plan, so that two methods can be compared section by section:

1. **Configuration**: the experiment, the method's hyperparameters, the settings shared by all
   methods. The values are the paper's.
2. **Data**: the labeled/unlabeled split, drawn from the seed exactly as the scripts draw it.
3. **Augmentations**: the weak and strong views.
4. **Model, EMA and optimiser**.
5. **The method**: its losses, one function per ingredient, then `train_step`, which computes the
   loss of one iteration.
6. **Evaluation and recording**.
7. **Training**: the same loop in every notebook, which calls `train_step`.
8. **Results**: the run against the paper's run of the same configuration.
9. **Things to try**: modifications worth making, with where to make them.

Everything is written out in the notebook, except the WideResNet architecture, which is imported
from `scripts/models.py` because it is the same for every method.

By default, `QUICK_TEST = True` stops the run after 1,000 iterations, about a minute on the GPU
used for the paper. Set it to `False` for a run of the paper.

## Where the runs go

A run writes `results/<dataset>-labeled-<n>-seed-<seed>/<method>_ema_notebook_metrics.json`, next
to the paper's `<method>_ema_metrics.json` and never over it (the `_notebook` suffix is `RUN_TAG`
in the configuration). `figures.ipynb` and `python scripts/run_analysis.py compare` both read it.
These files are ignored by git.

## Faithfulness to the scripts

At the default settings, each method notebook is the same computation as its script: same labeled
subset, same initial weights, same random draws. This was checked by running the first iteration of
each notebook and of its script. With `torch.backends.cudnn.benchmark = False` in both, the losses
are bit-for-bit identical. With it on, as in the paper, cuDNN picks its convolution algorithms by
timing them and may not pick the same ones in a notebook as in a script, which moves the last
digits. Beyond the first iteration, two runs drift apart slightly in any case, as two runs of the
same script do (see `results/README.md`).

The notebooks leave out the variants that no paper table needs from them: the FlexMatch- and
FreeMatch-style thresholds of `efficientmatch.py` (Table 4 runs them from the script) and
RegMixMatch's class-aware ResizeMix branch, which no run of the paper uses. Both remain in
`scripts/`.

The implementation details that the notebooks make visible (augmentations drawn per batch,
MixMatch's supervised term, RegMixMatch's warmup) are collected in
[docs/implementation_notes.md](../docs/implementation_notes.md).
