# Meilleure méthode par seuil d'accuracy réduit (-5% / -10%)

Ce document (généré par `scripts/best_method_by_threshold.py`) identifie, pour chacune des 4 configurations retenues de `EXPERIENCE_PRINCIPALE.md` (CIFAR-100 10000 labels exclu), quelle méthode parmi les 5 méthodes principales (efficientmatch_3, fixmatch, flexmatch, mixmatch, regmixmatch) atteint le plus vite (temps corrigé du coût des évaluations) et au moindre coût (FLOPs cumulés) un seuil d'accuracy réduit de 5 ou 10 points par rapport au target_acc habituel de la configuration. La variante efficientmatch_freematch n'est pas incluse.

Seuils (target original → -5% → -10%) :

| Configuration | Target original | Seuil -5% | Seuil -10% |
|---|---:|---:|---:|
| SVHN, 250 labels | 90% | 85% | 80% |
| CIFAR-10, 250 labels | 80% | 75% | 70% |
| CIFAR-10, 4000 labels | 90% | 85% | 80% |
| CIFAR-100, 2500 labels (WF4) | 50% | 45% | 40% |

## Tableau récapitulatif (méthode gagnante, majorité sur 3 seeds)

| Configuration | -5% (temps) | -5% (FLOPs) | -10% (temps) | -10% (FLOPs) |
|---|---|---|---|---|
| SVHN, 250 labels | efficientmatch_3 | pas de majorité | mixmatch | mixmatch |
| CIFAR-10, 250 labels | efficientmatch_3 | efficientmatch_3 | efficientmatch_3 | mixmatch |
| CIFAR-10, 4000 labels | mixmatch | mixmatch | mixmatch | mixmatch |
| CIFAR-100, 2500 labels (WF4) | pas de majorité | pas de majorité | mixmatch | mixmatch |

## Détail par seed

### SVHN, 250 labels

| Seuil | Seed | Meilleur temps | Meilleur FLOPs | Temps corrigé (min) par méthode | TFLOPs par méthode |
|---|---|---|---|---|---|
| -5% (85%) | 2312 | mixmatch | mixmatch | efficientmatch_3 3.8, fixmatch 4.5, flexmatch n/a, mixmatch 3.7, regmixmatch 7.4 | efficientmatch_3 3 332, fixmatch 3 825, flexmatch n/a, mixmatch 3 016, regmixmatch 7 567 |
| -5% (85%) | 308 | efficientmatch_3 | efficientmatch_3 | efficientmatch_3 3.4, fixmatch 3.6, flexmatch n/a, mixmatch 10.3, regmixmatch 6.4 | efficientmatch_3 2 961, fixmatch 2 975, flexmatch n/a, mixmatch 9 200, regmixmatch 6 622 |
| -5% (85%) | 2701 | efficientmatch_3 | fixmatch | efficientmatch_3 4.9, fixmatch 5.0, flexmatch 31.2, mixmatch 11.5, regmixmatch 8.5 | efficientmatch_3 4 442, fixmatch 4 250, flexmatch 25 928, mixmatch 10 256, regmixmatch 8 513 |
| -10% (80%) | 2312 | mixmatch | mixmatch | efficientmatch_3 3.1, fixmatch 3.6, flexmatch 20.5, mixmatch 2.8, regmixmatch 5.6 | efficientmatch_3 2 591, fixmatch 2 975, flexmatch 17 002, mixmatch 2 111, regmixmatch 5 676 |
| -10% (80%) | 308 | mixmatch | mixmatch | efficientmatch_3 2.7, fixmatch 3.2, flexmatch 37.0, mixmatch 2.4, regmixmatch 5.5 | efficientmatch_3 2 221, fixmatch 2 550, flexmatch 31 029, mixmatch 1 810, regmixmatch 5 676 |
| -10% (80%) | 2701 | fixmatch | fixmatch | efficientmatch_3 4.1, fixmatch 4.1, flexmatch 17.6, mixmatch 4.4, regmixmatch 7.6 | efficientmatch_3 3 702, fixmatch 3 400, flexmatch 14 452, mixmatch 3 620, regmixmatch 7 567 |

### CIFAR-10, 250 labels

| Seuil | Seed | Meilleur temps | Meilleur FLOPs | Temps corrigé (min) par méthode | TFLOPs par méthode |
|---|---|---|---|---|---|
| -5% (75%) | 2312 | efficientmatch_3 | efficientmatch_3 | efficientmatch_3 15.0, fixmatch 34.9, flexmatch 29.8, mixmatch 59.2, regmixmatch 30.3 | efficientmatch_3 15 547, fixmatch 32 729, flexmatch 25 928, mixmatch 19 003, regmixmatch 33 108 |
| -5% (75%) | 308 | efficientmatch_3 | efficientmatch_3 | efficientmatch_3 10.6, fixmatch 22.2, flexmatch 23.3, mixmatch 19.6, regmixmatch 26.7 | efficientmatch_3 10 735, fixmatch 20 827, flexmatch 19 977, mixmatch 19 607, regmixmatch 29 324 |
| -5% (75%) | 2701 | efficientmatch_3 | efficientmatch_3 | efficientmatch_3 10.9, fixmatch 22.8, flexmatch 18.0, mixmatch 11.8, regmixmatch 25.9 | efficientmatch_3 11 105, fixmatch 21 252, flexmatch 15 302, mixmatch 11 613, regmixmatch 28 378 |
| -10% (70%) | 2312 | efficientmatch_3 | mixmatch | efficientmatch_3 9.2, fixmatch 19.3, flexmatch 18.7, mixmatch 26.0, regmixmatch 20.1 | efficientmatch_3 9 254, fixmatch 17 852, flexmatch 16 152, mixmatch 8 144, regmixmatch 21 756 |
| -10% (70%) | 308 | efficientmatch_3 | mixmatch | efficientmatch_3 6.8, fixmatch 13.9, flexmatch 12.6, mixmatch 6.8, regmixmatch 16.4 | efficientmatch_3 6 663, fixmatch 12 752, flexmatch 10 626, mixmatch 6 485, regmixmatch 17 973 |
| -10% (70%) | 2701 | mixmatch | mixmatch | efficientmatch_3 7.4, fixmatch 14.0, flexmatch 12.2, mixmatch 5.2, regmixmatch 15.6 | efficientmatch_3 7 404, fixmatch 13 177, flexmatch 10 201, mixmatch 4 826, regmixmatch 17 027 |

### CIFAR-10, 4000 labels

| Seuil | Seed | Meilleur temps | Meilleur FLOPs | Temps corrigé (min) par méthode | TFLOPs par méthode |
|---|---|---|---|---|---|
| -5% (85%) | 2312 | mixmatch | mixmatch | efficientmatch_3 6.9, fixmatch 12.0, flexmatch 12.0, mixmatch 3.8, regmixmatch 12.2 | efficientmatch_3 7 033, fixmatch 11 476, flexmatch 10 201, mixmatch 3 620, regmixmatch 13 243 |
| -5% (85%) | 308 | mixmatch | mixmatch | efficientmatch_3 7.4, fixmatch 11.2, flexmatch 11.5, mixmatch 4.1, regmixmatch 11.3 | efficientmatch_3 7 404, fixmatch 10 626, flexmatch 9 776, mixmatch 3 921, regmixmatch 12 297 |
| -5% (85%) | 2701 | mixmatch | mixmatch | efficientmatch_3 7.2, fixmatch 12.1, flexmatch 10.1, mixmatch 3.8, regmixmatch 13.0 | efficientmatch_3 7 404, fixmatch 11 476, flexmatch 8 926, mixmatch 3 620, regmixmatch 14 189 |
| -10% (80%) | 2312 | mixmatch | mixmatch | efficientmatch_3 3.5, fixmatch 5.2, flexmatch 6.2, mixmatch 1.9, regmixmatch 6.2 | efficientmatch_3 3 332, fixmatch 4 676, flexmatch 5 101, mixmatch 1 508, regmixmatch 6 622 |
| -10% (80%) | 308 | mixmatch | mixmatch | efficientmatch_3 3.5, fixmatch 4.8, flexmatch 5.3, mixmatch 1.9, regmixmatch 6.2 | efficientmatch_3 3 332, fixmatch 4 250, flexmatch 4 250, mixmatch 1 508, regmixmatch 6 622 |
| -10% (80%) | 2701 | mixmatch | mixmatch | efficientmatch_3 3.8, fixmatch 5.2, flexmatch 5.1, mixmatch 2.0, regmixmatch 7.0 | efficientmatch_3 3 702, fixmatch 4 676, flexmatch 4 250, mixmatch 1 659, regmixmatch 7 567 |

### CIFAR-100, 2500 labels (WF4)

| Seuil | Seed | Meilleur temps | Meilleur FLOPs | Temps corrigé (min) par méthode | TFLOPs par méthode |
|---|---|---|---|---|---|
| -5% (45%) | 2312 | mixmatch | mixmatch | efficientmatch_3 36.8, fixmatch 97.2, flexmatch 40.8, mixmatch 35.0, regmixmatch 40.1 | efficientmatch_3 59 093, fixmatch 150 970, flexmatch 62 065, mixmatch 54 760, regmixmatch 85 871 |
| -5% (45%) | 308 | efficientmatch_3 | efficientmatch_3 | efficientmatch_3 33.8, fixmatch 80.1, flexmatch 46.1, mixmatch 43.0, regmixmatch 35.1 | efficientmatch_3 54 056, fixmatch 124 131, flexmatch 70 452, mixmatch 69 045, regmixmatch 74 670 |
| -5% (45%) | 2701 | flexmatch | flexmatch | efficientmatch_3 35.6, fixmatch 88.9, flexmatch 34.8, mixmatch 49.7, regmixmatch 42.1 | efficientmatch_3 56 978, fixmatch 139 228, flexmatch 53 678, mixmatch 79 759, regmixmatch 89 604 |
| -10% (40%) | 2312 | mixmatch | mixmatch | efficientmatch_3 22.2, fixmatch 54.6, flexmatch 28.9, mixmatch 11.0, regmixmatch 29.7 | efficientmatch_3 35 157, fixmatch 83 872, flexmatch 43 613, mixmatch 16 666, regmixmatch 63 470 |
| -10% (40%) | 308 | mixmatch | mixmatch | efficientmatch_3 23.9, fixmatch 41.1, flexmatch 32.8, mixmatch 11.1, regmixmatch 24.6 | efficientmatch_3 37 985, fixmatch 63 743, flexmatch 50 323, mixmatch 17 261, regmixmatch 52 269 |
| -10% (40%) | 2701 | mixmatch | mixmatch | efficientmatch_3 23.0, fixmatch 53.7, flexmatch 27.3, mixmatch 12.1, regmixmatch 29.9 | efficientmatch_3 36 524, fixmatch 83 872, flexmatch 41 936, mixmatch 18 452, regmixmatch 63 470 |

`n/a` : seuil jamais atteint par la run.

## Reproduire

```bash
python scripts/best_method_by_threshold.py
```
