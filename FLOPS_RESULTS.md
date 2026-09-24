# Résultats de l'analyse FLOPs par itération

Mesures réalisées via `scripts/flops_analysis.py` sur cuda (`torch==2.12.1+cu130`, Python 3.11.15), avec `torch.utils.flop_counter.FlopCounterMode` sur des tenseurs factices (dummy tensors) -- indépendant des données, de `--optimized`/`--amp`/`torch.compile`.

Batch labellisé = 64 (toutes architectures). CIFAR-10 et SVHN utilisent WideResNet-28-2 dans tous
les scripts de ce dépôt ; CIFAR-100 utilisait WideResNet-28-8 (`--widen_factor 8`) à l'origine, mais
**WRN-28-8 sature la VRAM de ce GPU (8GB) avec certaines méthodes** (voir `RUNS_TO_REVISIT.md`,
incident regmixmatch/CIFAR-100) -- WideResNet-28-4 (`--widen_factor 4`) est donc devenu l'architecture
de facto pour la plupart des runs CIFAR-100 de cette session (fixmatch, flexmatch, mixmatch,
regmixmatch, efficientmatch_3 ont tous été retestés en WF4). Les FLOPs/itération sont spécifiques à
l'architecture -- ne jamais réutiliser une valeur d'une architecture pour une autre. Les fichiers de
résultats distinguent l'architecture par un suffixe `_wf{N}` dans leur nom (ex.
`fixmatch_ema_wf4_metrics.json`) ; un fichier sans suffixe sur CIFAR-100 est un run historique en
WRN-28-8 (avant l'ajout de cette convention de nommage).

## Résultats — WideResNet-28-2 (CIFAR-10, SVHN)

| Méthode | mu | Batch non labellisé | FLOPs / itération | GFLOPs / itération |
|---|---:|---:|---:|---:|
| efficientmatch | 3 | 192 | 7.404e+11 | 740.35 |
| efficientmatch_2 | 3 | 192 | 7.404e+11 | 740.35 |
| efficientmatch_3 | 3 | 192 | 7.404e+11 | 740.35 |
| efficientmatch_freematch (CIFAR-10) | 3 | 192 | 7.404e+11 | 740.35 |
| efficientmatch_freematch (SVHN, borne active) | 3 | 192 | 7.404e+11 | 740.35 |
| efficientmatch_3_mu1 | 1 | 64 | 3.565e+11 | 356.46 |
| efficientmatch_3_mu5 | 5 | 320 | 1.124e+12 | 1124.25 |
| efficientmatch_3_mu7 | 7 | 448 | 1.508e+12 | 1508.14 |
| efficientmatch_flex_mu2 | 2 | 128 | 5.484e+11 | 548.40 |
| fixmatch | 7 | 448 | 8.501e+11 | 850.10 |
| flexmatch | 7 | 448 | 8.501e+11 | 850.10 |
| mixmatch | 1 | 64 | 3.016e+11 | 301.64 |
| regmixmatch | 7 | 448 | 1.892e+12 | 1891.86 |
| regmixmatch_mu3 | 3 | 192 | 9.048e+11 | 904.80 |
| sequencematch | 7 | 448 | 1.810e+12 | 1809.61 |

**efficientmatch_freematch (seuillage adaptatif FreeMatch) — valeurs exactes.** Coût du modèle
(`FlopCounterMode`, conv/matmul uniquement) identique à efficientmatch_3, plus les opérations
élémentaires du seuillage (moyenne, deux EMA, max, produit, comparaisons ; 1 FLOP par opération,
indexation et copies = 0), comptées à la main dans `run_analysis.freematch_threshold_flops` :

| Config | FLOPs / itération exacts | Surcoût vs efficientmatch_3 (740 351 508 480) |
|---|---:|---:|
| CIFAR-10 (10 classes) | 740 351 510 836 | +2 356 (3.2e-9 relatif) |
| SVHN (10 classes, borne [0.9, 0.95] active) | 740 351 511 220 | +2 740 (3.7e-9 relatif) |

Le surcoût est invisible dans les tableaux arrondis ci-dessus. Mesuré via
`python flops_analysis.py --methods efficientmatch_3 efficientmatch_freematch efficientmatch_freematch_svhn --widen-factor 2`.

## Résultats — WideResNet-28-4 (CIFAR-100, architecture de facto sur ce GPU)

Mesuré via `python flops_analysis.py --methods fixmatch flexmatch mixmatch regmixmatch efficientmatch_3 --widen-factor 4`.

| Méthode | mu | Batch non labellisé | FLOPs / itération | GFLOPs / itération |
|---|---:|---:|---:|---:|
| fixmatch | 7 | 448 | 3.355e+12 | 3354.88 |
| flexmatch | 7 | 448 | 3.355e+12 | 3354.88 |
| mixmatch | 1 | 64 | 1.190e+12 | 1190.43 |
| regmixmatch | 7 | 448 | 7.467e+12 | 7467.01 |
| efficientmatch_3 | 3 | 192 | 2.922e+12 | 2921.93 |
| efficientmatch_freematch (CIFAR-100) | 3 | 192 | 2.922e+12 | 2921.93 |

Valeurs exactes pour efficientmatch_freematch en CIFAR-100 (100 classes) : 2 921 930 903 158 FLOPs/it,
soit +20 086 FLOPs/it (6.9e-9 relatif) par rapport à efficientmatch_3 (2 921 930 883 072). Mesuré via
`python flops_analysis.py --methods efficientmatch_3 efficientmatch_freematch_c100 --widen-factor 4`.

## Résultats — WideResNet-28-8 (CIFAR-100, runs historiques uniquement -- voir avertissement VRAM ci-dessus)

Mesuré via `python flops_analysis.py --methods fixmatch flexmatch mixmatch --widen-factor 8`
(seules méthodes testées sur CIFAR-100 à ce jour, cf. `SEEDS_2312_308_2701_INVENTORY.md`).

| Méthode | mu | Batch non labellisé | FLOPs / itération | GFLOPs / itération |
|---|---:|---:|---:|---:|
| fixmatch | 7 | 448 | 1.333e+13 | 13332.36 |
| flexmatch | 7 | 448 | 1.333e+13 | 13332.36 |
| mixmatch | 1 | 64 | 4.731e+12 | 4730.83 |

## Coût d'une évaluation (indépendant de la méthode SSL)

Toutes les méthodes appellent la même fonction `evaluate_f1_and_accuracy()` (`scripts/utils.py`)
sur le même type de modèle/test_loader à chaque `test_period` steps -- le coût d'une évaluation ne
dépend donc que de (dataset, architecture), pas de la méthode SSL. Mesuré via
`scripts/measure_eval_time.py` (isolé, EMA-copy inclus) et vérifié en conditions réelles via
`scripts/ghost_method.py` (mêmes dataloaders/augmentations/EMA/torch.compile qu'un vrai run, mais
la boucle d'entraînement ne fait rien -- seule l'évaluation consomme du GPU) :

| Architecture | Dataset test | Mesure isolée | Mesure in situ | Écart |
|---|---|---:|---:|---:|
| WRN-28-2 | CIFAR-10 (10 000 images) | 544.7 ms | 553.7 ms | +1.6% |
| WRN-28-4 | CIFAR-100 (10 000 images) | 1 492.8 ms | 1 499.1 ms | +0.4% |
| WRN-28-8 | CIFAR-100 (10 000 images) | 4 434.8 ms | 4 476.3 ms | +0.9% |

L'accord à moins de 2% entre les deux méthodes de mesure confirme que la valeur isolée est fiable
pour estimer le coût réel pendant un run d'entraînement.

**Impact sur le temps de run mesuré**, pour le run ayant le plus grand nombre d'évaluations
(donc le cas le plus défavorable) sur chaque dataset :

| Dataset | Run (le plus d'évaluations) | n_evals | Temps total | Overhead éval | % du temps total |
|---|---|---:|---:|---:|---:|
| CIFAR-10 (WRN-28-2) | mixmatch, 250 labels, seed 308 | 1 767 | 276.6 min | 16.31 min | **5.89%** |
| CIFAR-100 (WRN-28-8) | fixmatch, 250 labels, seed 2312 | 90 | 459.3 min | 6.71 min | **1.46%** |

Même dans le pire cas (méthode la plus lente à converger, donc avec le plus d'évaluations),
l'overhead d'évaluation reste modéré : sous 6% sur CIFAR-10, sous 1.5% sur CIFAR-100 malgré un coût
par évaluation ~8x plus élevé (le run cifar100 étant beaucoup plus long en absolu, l'overhead pèse
proportionnellement moins). Pour les comparaisons de vitesse entre méthodes sur une même config,
cet overhead n'est donc généralement pas un facteur de confusion significatif -- sauf pour des runs
très courts avec beaucoup d'évaluations rapprochées (`test_period` faible), où il peut représenter
jusqu'à ~10% du temps mesuré (cf. cas cifar100-10000/mixmatch, 11.26% observé pour un run de 28 min
avec 43 évaluations).
