# Inventaire complet — seeds 2312, 0308, 2701 (CIFAR-10 et SVHN)

Toutes les expériences disponibles dans `results/` pour les seeds 2312, 0308 et 2701, sur CIFAR-10
et SVHN, tous budgets de labels confondus. FLOPs/itération (indépendants du hardware, mesurés via
`scripts/flops_analysis.py`) : fixmatch/flexmatch = 850.10 GFLOPs (mu=7), mixmatch = 301.64 GFLOPs
(mu=1), efficientmatch/efficientmatch_2/efficientmatch_flex = 740.35 GFLOPs (mu=3, efficientmatch_2
non mesuré séparément — valeur approximée par efficientmatch v1, même architecture). Les variantes
`_mu2` (mu=2) n'ont pas de FLOPs/it mesurés séparément.

Note générale sur les seuils cibles : SVHN 250 labels visait 90% (sauf mentions contraires),
CIFAR-10 250 labels visait 80%, CIFAR-10 4000 labels visait 90%. SVHN 1000/40 labels et les
variantes `_flex`/`efficientmatch` (v1) sur SVHN 250 n'avaient pas toujours de `target_acc` fixé
ou visaient un seuil différent — voir la colonne Acc pour la valeur réellement atteinte.

---

## CIFAR-10, 250 labels (target 80%)

### Seed 2312

| Méthode | Steps | Temps | Acc | FLOPs |
|---|---:|---:|---:|---:|
| efficientmatch_2 | 45 000 | 33.43 min | 80.26% | 33 315.8 TFLOPs |
| efficientmatch_flex | 47 000 | 34.79 min | 80.21% | 34 796.4 TFLOPs |
| efficientmatch_flex_mu2 | 88 000 | 50.68 min | 80.33% | N/A (mu=2) |
| efficientmatch (v1) | 82 000 | 58.99 min | 80.10% | 60 708.7 TFLOPs |
| flexmatch | 70 500 | 114.07 min | 80.17% | 59 932.1 TFLOPs |
| fixmatch | 170 500 | 272.46 min | 80.06% | 144 942.0 TFLOPs |
| mixmatch | 828 500 | 779.67 min (13h) | 78.42% (max, **jamais atteint 80%**) | 249 908.7 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Acc | FLOPs |
|---|---:|---:|---:|---:|
| efficientmatch_2 | 55 500 | 40.16 min | 80.35% | 41 089.4 TFLOPs |
| flexmatch | 49 500 | 45.21 min | 80.00% | 42 079.9 TFLOPs |
| fixmatch | 64 000 | 59.43 min | 80.00% | 54 406.4 TFLOPs |
| mixmatch | 883 000 | 276.64 min (4h37) | 80.07% | 266 348.1 TFLOPs |

*(efficientmatch v1 non testé sur cette seed — volontairement exclu sur demande utilisateur après
le run seed 2312)*

### Seed 2701

| Méthode | Steps | Temps | Acc | FLOPs |
|---|---:|---:|---:|---:|
| efficientmatch_2 | 34 000 | 24.38 min | 80.00% | 25 171.9 TFLOPs |
| flexmatch | 33 000 | 31.35 min | 80.19% | 28 053.3 TFLOPs |
| fixmatch | 61 500 | 57.42 min | 80.15% | 52 281.2 TFLOPs |
| mixmatch | 464 500 | 145.59 min (2h26) | 80.10% | 140 111.8 TFLOPs |

**Synthèse CIFAR-10/250 labels** : classement instable d'une seed à l'autre (efficientmatch_2
toujours rapide et dans le haut du classement, mais mixmatch alterne entre très lent — seed 2312 :
n'atteint jamais 80% en 13h — et beaucoup plus rapide sur les 2 autres seeds tout en restant le
plus gros consommateur de FLOPs). fixmatch est systématiquement le plus lent des méthodes qui
atteignent le seuil. Détail complet dans `EXPERIMENT_LOG.md` §4.

---

## CIFAR-10, 4000 labels (target 90%)

### Seed 2312

| Méthode | Steps | Temps | Acc | FLOPs |
|---|---:|---:|---:|---:|
| efficientmatch_2 | 27 500 | 19.55 min | 90.01% | 20 359.6 TFLOPs |
| efficientmatch_flex | 63 500 | 45.59 min | 90.97% (max) | 47 012.2 TFLOPs |
| mixmatch | 198 500 | 59.20 min | 90.01% | 59 875.5 TFLOPs |
| flexmatch | 81 500 | 73.65 min | 90.03% | 69 283.1 TFLOPs |
| fixmatch | 99 000 | 87.06 min | 90.03% | 84 159.9 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Acc | FLOPs |
|---|---:|---:|---:|---:|
| efficientmatch_2 | 32 500 | 23.02 min | 90.03% | 24 061.4 TFLOPs |
| mixmatch | 213 500 | 63.57 min | 90.05% | 64 400.1 TFLOPs |
| flexmatch | 84 000 | 75.40 min | 90.06% | 71 408.4 TFLOPs |
| fixmatch | 91 500 | 80.79 min | 90.02% | 77 784.1 TFLOPs |

### Seed 2701

| Méthode | Steps | Temps | Acc | FLOPs |
|---|---:|---:|---:|---:|
| efficientmatch_2 | 36 000 | 25.43 min | 90.00% | 26 652.6 TFLOPs |
| mixmatch | 165 500 | 49.34 min | 90.04% | 49 921.4 TFLOPs |
| flexmatch | 84 000 | 75.42 min | 90.03% | 71 408.4 TFLOPs |
| fixmatch | 100 500 | 88.80 min | 90.15% | 85 435.1 TFLOPs |

**Synthèse CIFAR-10/4000 labels** : classement **parfaitement stable sur les 3 seeds** :
efficientmatch_2 > mixmatch > flexmatch > fixmatch, en temps comme en FLOPs. efficientmatch_2
gagne toujours nettement (2.4x-4.5x plus rapide que le 2e, mixmatch). mixmatch, catastrophique à
250 labels, redevient compétitif ici (converge en moins d'1h sur les 3 seeds). Détail dans
`CIFAR10_4000LABELS_RESULTS.md`.

---

## SVHN, 250 labels (target 90%)

### Seed 2312

| Méthode | Steps | Temps | Acc | FLOPs |
|---|---:|---:|---:|---:|
| mixmatch | 18 500 | 6.81 min | 90.27% | **5 580.3 TFLOPs** |
| efficientmatch_2 | 8 500 | **6.76 min** | 90.04% | 6 293.0 TFLOPs |
| fixmatch | 8 000 | 7.82 min | 90.01% | 6 800.8 TFLOPs |
| efficientmatch (v1) | 34 000 | 28.53 min | 90.05% | 25 171.9 TFLOPs |
| efficientmatch_flex | 8 500 | 7.26 min | 77.44% (max, jamais 90%) | 6 293.0 TFLOPs |
| efficientmatch_2_flex | 13 000 | 10.49 min | 79.76% (max, jamais 90%) | 9 624.5 TFLOPs |
| flexmatch | 104 000 | 97.51 min | 83.54% (max, **jamais atteint 90%**, arrêté manuellement) | 88 410.4 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Acc | FLOPs |
|---|---:|---:|---:|---:|
| fixmatch | 6 500 | **6.44 min** | 90.07% | **5 525.6 TFLOPs** |
| efficientmatch_2 | 8 500 | 6.76 min | 90.24% | 6 293.0 TFLOPs |
| mixmatch | 52 500 | 18.38 min | 91.06% | 15 836.1 TFLOPs |
| flexmatch | 30 000 | 28.69 min | 82.38% (max, **jamais atteint 90%**, arrêté manuellement) | 25 503.0 TFLOPs |
| efficientmatch (v1) | 1 | 0.46 min | 8.03% (killé volontairement, run non exploitable) | — |

### Seed 2701

| Méthode | Steps | Temps | Acc | FLOPs |
|---|---:|---:|---:|---:|
| efficientmatch_2 | 10 000 | 7.86 min | 90.02% | 7 403.5 TFLOPs |
| fixmatch | 8 000 | 7.90 min | 90.18% | 6 800.8 TFLOPs |
| mixmatch | 35 000 | 12.44 min | 90.82% | 10 557.4 TFLOPs |
| flexmatch | 40 000 | 37.99 min | 85.10% (max, **jamais atteint 90%**, arrêté au timeout 30 min) | 34 004.0 TFLOPs |

**Synthèse SVHN/250 labels** : **flexmatch échoue sur les 3 seeds** (plafond 82-85%, jamais 90%) —
problème méthodologique reproductible documenté dans `RUNS_TO_REVISIT.md`, pas de la variance.
Parmi les méthodes qui réussissent, le classement temps/FLOPs varie d'une seed à l'autre entre
efficientmatch_2, fixmatch et mixmatch (jamais le même vainqueur deux fois), mais les trois restent
toujours regroupées dans une fourchette étroite (6-19 min) contrairement à flexmatch.
efficientmatch (v1) et ses variantes `_flex` n'ont été testées que ponctuellement, pas sur les 3
seeds de façon systématique.

---

## SVHN, autres budgets de labels (seed 2312 uniquement, pas de comparaison multi-seed)

| Config | Méthode | Steps | Temps | Acc |
|---|---|---:|---:|---:|
| 1000 labels | efficientmatch (v1) | 84 500 | 61.36 min | 92.99% (max) |
| 1000 labels | fixmatch | 105 000 | 98.23 min | 93.89% (max) |
| 40 labels | efficientmatch (v1) | 82 000 | 59.75 min | 56.23% (max) |

Ces trois runs n'existent que pour la seed 2312 — pas de données comparables sur 0308/2701 pour
ces budgets de labels.

---

## Vue d'ensemble : ce qui a et n'a pas été fait sur les 3 seeds

| Config | efficientmatch_2 | fixmatch | flexmatch | mixmatch | efficientmatch (v1) |
|---|:---:|:---:|:---:|:---:|:---:|
| CIFAR-10, 250 labels | ✅ 3/3 seeds | ✅ 3/3 | ✅ 3/3 (échoue jamais mais lent) | ✅ 3/3 (1 jamais convergé) | seed 2312 seulement |
| CIFAR-10, 4000 labels | ✅ 3/3 seeds | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ❌ aucune |
| SVHN, 250 labels | ✅ 3/3 seeds | ✅ 3/3 | ✅ 3/3 (**0/3 atteint 90%**) | ✅ 3/3 | seed 2312 (+ tentative avortée seed 0308) |
| SVHN, 1000/40 labels | ❌ aucune | seed 2312 (1000 seulement) | ❌ aucune | ❌ aucune | seed 2312 seulement |

**Couverture complète et comparable sur les 3 seeds** : seulement CIFAR-10 250 labels, CIFAR-10
4000 labels, et SVHN 250 labels, pour les 4 méthodes efficientmatch_2/fixmatch/flexmatch/mixmatch.
efficientmatch (v1) n'a jamais été testé de façon systématique sur les 3 seeds pour aucune config.
SVHN à d'autres budgets de labels (1000, 40) reste à seed unique.
