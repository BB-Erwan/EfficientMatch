# Résultats de l'analyse FLOPs par itération

Mesures réalisées via `scripts/flops_analysis.py` sur cuda (`torch==2.12.1+cu130`, Python 3.11.15), avec `torch.utils.flop_counter.FlopCounterMode` sur des tenseurs factices (dummy tensors) -- indépendant des données, de `--optimized`/`--amp`/`torch.compile`.

Modèle : WideResNet-28-2 ; batch labellisé = 64.

## Résultats

| Méthode | mu | Batch non labellisé | FLOPs / itération | GFLOPs / itération |
|---|---:|---:|---:|---:|
| efficientmatch | 3 | 192 | 7.404e+11 | 740.35 |
| fixmatch | 7 | 448 | 8.501e+11 | 850.10 |
| flexmatch | 7 | 448 | 8.501e+11 | 850.10 |
| mixmatch | 1 | 64 | 3.016e+11 | 301.64 |
| sequencematch | 7 | 448 | 1.810e+12 | 1809.61 |
