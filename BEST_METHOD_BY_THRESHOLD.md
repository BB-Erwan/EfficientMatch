# Meilleure méthode par seuil d'accuracy réduit (-5% / -10%)

Ce document identifie, pour chacune des 4 configurations de `EXPERIENCE_PRINCIPALE.md`, quelle
méthode parmi les 5 méthodes principales (efficientmatch_3, fixmatch, flexmatch, mixmatch,
regmixmatch) atteint le plus vite (temps corrigé) et au moindre coût (FLOPs totaux) un seuil
d'accuracy réduit de 5 ou 10 points par rapport au target_acc habituel de chaque configuration.
Chaque cellule ci-dessous contient uniquement le nom de la méthode gagnante ; le détail par seed est
donné plus bas.

Seuils utilisés (target_acc original → -5% → -10%) :

| Configuration | Target original | Seuil -5% | Seuil -10% |
|---|---:|---:|---:|
| SVHN, 250 labels | 90% | 85% | 80% |
| CIFAR-10, 250 labels | 80% | 75% | 70% |
| CIFAR-10, 4000 labels | 90% | 85% | 80% |
| CIFAR-100, 10000 labels (WF4) | 60% | 55% | 50% |

## Tableau récapitulatif (méthode gagnante, majorité sur 3 seeds)

| Configuration | -5% (temps) | -5% (FLOPs) | -10% (temps) | -10% (FLOPs) |
|---|---|---|---|---|
| SVHN, 250 labels | efficientmatch_3 | pas de majorité (3 méthodes différentes) | mixmatch | mixmatch |
| CIFAR-10, 250 labels | efficientmatch_3 | efficientmatch_3 | efficientmatch_3 | mixmatch |
| CIFAR-10, 4000 labels | mixmatch | mixmatch | mixmatch | mixmatch |
| CIFAR-100, 10000 labels | mixmatch | mixmatch | mixmatch | mixmatch |

## Détail par seed

### SVHN, 250 labels

| Seuil | Seed | Meilleur temps | Meilleur FLOPs |
|---|---|---|---|
| -5% (85%) | 2312 | mixmatch | mixmatch |
| -5% (85%) | 308 | efficientmatch_3 | efficientmatch_3 |
| -5% (85%) | 2701 | efficientmatch_3 | fixmatch |
| -10% (80%) | 2312 | mixmatch | mixmatch |
| -10% (80%) | 308 | mixmatch | mixmatch |
| -10% (80%) | 2701 | fixmatch | fixmatch |

### CIFAR-10, 250 labels

| Seuil | Seed | Meilleur temps | Meilleur FLOPs |
|---|---|---|---|
| -5% (75%) | 2312 | efficientmatch_3 | efficientmatch_3 |
| -5% (75%) | 308 | efficientmatch_3 | efficientmatch_3 |
| -5% (75%) | 2701 | efficientmatch_3 | efficientmatch_3 |
| -10% (70%) | 2312 | efficientmatch_3 | mixmatch |
| -10% (70%) | 308 | efficientmatch_3 | mixmatch |
| -10% (70%) | 2701 | mixmatch | mixmatch |

### CIFAR-10, 4000 labels

| Seuil | Seed | Meilleur temps | Meilleur FLOPs |
|---|---|---|---|
| -5% (85%) | 2312 | mixmatch | mixmatch |
| -5% (85%) | 308 | mixmatch | mixmatch |
| -5% (85%) | 2701 | mixmatch | mixmatch |
| -10% (80%) | 2312 | mixmatch | mixmatch |
| -10% (80%) | 308 | mixmatch | mixmatch |
| -10% (80%) | 2701 | mixmatch | mixmatch |

### CIFAR-100, 10000 labels (WF4)

| Seuil | Seed | Meilleur temps | Meilleur FLOPs |
|---|---|---|---|
| -5% (55%) | 2312 | mixmatch | mixmatch |
| -5% (55%) | 308 | mixmatch | mixmatch |
| -5% (55%) | 2701 | mixmatch | mixmatch |
| -10% (50%) | 2312 | mixmatch | mixmatch |
| -10% (50%) | 308 | mixmatch | mixmatch |
| -10% (50%) | 2701 | mixmatch | mixmatch |

## Interprétation

- **mixmatch domine largement dès qu'on réduit le seuil d'accuracy**, et sans partage sur CIFAR-10
  4000 labels et CIFAR-100 10000 labels (6/6 seeds pour les deux seuils sur les deux
  configurations). Son coût par itération très faible (301.64 GFLOPs en WRN-28-2, 1 190.43 en
  WRN-28-4 — 2 à 3x moins que les autres méthodes) lui permet d'atteindre rapidement un seuil
  d'accuracy modeste, même s'il est ensuite beaucoup plus lent que les autres méthodes pour
  atteindre le target_acc complet (cf. `EXPERIENCE_PRINCIPALE.md`).
- **efficientmatch_3 ne l'emporte que sur CIFAR-10 250 labels et partiellement sur SVHN 250
  labels**, c'est-à-dire les configurations à très peu de labels où son canal Mixup apporte le plus
  de valeur (cf. `ABLATION_SOFT_VS_HARD_EFFICIENTMATCH.md`) — cohérent avec la conclusion selon
  laquelle son avantage d'efficacité est concentré sur les régimes à faible nombre de labels.
  Dès que le seuil descend à -10% sur CIFAR-10 250, mixmatch reprend l'avantage en FLOPs (mais pas
  toujours en temps).
- **Aucune méthode ne l'emporte de façon univoque sur SVHN 250 labels à -5%** : les 3 seeds
  donnent 3 réponses différentes en FLOPs (mixmatch, efficientmatch_3, fixmatch), signe que les
  méthodes sont très proches à ce seuil précis sur cette configuration.
- **Attention à l'interprétation** : "meilleur à un seuil réduit" ne veut pas dire "meilleure
  méthode globalement" — mixmatch est nettement plus lent que les autres méthodes pour atteindre le
  target_acc complet de chaque configuration (voir les tableaux complets dans
  `EXPERIENCE_PRINCIPALE.md`). Ce document répond uniquement à la question "quelle méthode est la
  plus rapide/économe pour un niveau de qualité intermédiaire donné".

## Reproduire

```bash
python scripts/run_analysis.py at-acc --dataset <dataset> --num_labeled <N> --seed <seed> --target <seuil>
```

Exemple : `python scripts/run_analysis.py at-acc --dataset cifar10 --num_labeled 250 --seed 2312 --target 0.75`
