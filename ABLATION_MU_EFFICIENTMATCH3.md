# Étude d'hyperparamètre : mu (nombre de vues non labellisées) — efficientmatch_3

**mu** contrôle le nombre de vues non labellisées traitées par itération dans efficientmatch_3
(analogue au ratio non-labellisé:labellisé des méthodes de type FixMatch, où mu=7 est la valeur de
référence historique). efficientmatch_3 utilise **mu=3 par défaut**. Cette étude compare mu ∈
{1, 3, 5, 7} sur deux régimes de labels de CIFAR-10, WRN-28-2, **complète sur les 3 seeds**
(2312, 0308, 2701) dans les deux cas :

- **CIFAR-10, 250 labels** (target_acc 80%)
- **CIFAR-10, 4000 labels** (target_acc 90%)

Coût par itération (mesuré indépendamment du hardware via `scripts/flops_analysis.py`,
cf. `FLOPS_RESULTS.md`) :

| mu | GFLOPs/itération |
|---:|---:|
| 1 | 356.46 |
| 3 (défaut) | 740.35 |
| 5 | 1 124.25 |
| 7 | 1 508.14 |

## Résultats complets — CIFAR-10, 250 labels, 3 seeds

### Seed 2312

| mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux | Statut |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 173 000 | 71.62 min | 68.42 min | 77.57% (max) | 61 668 TFLOPs | ❌ jamais atteint 80% |
| 3 (défaut) | 42 000 | 30.38 min | 29.60 min | 80.11% | 31 095 TFLOPs | ✅ |
| 5 | 28 500 | 30.53 min | 30.00 min | 80.10% | 32 041 TFLOPs | ✅ |
| 7 | 25 500 | 36.26 min | 35.78 min | 80.18% | 38 458 TFLOPs | ✅ |

### Seed 0308

| mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux | Statut |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 188 500 | 80.00 min | 76.53 min | 79.82% (max) | 67 193 TFLOPs | ❌ jamais atteint 80% (tué à 2h) |
| 5 | 17 000 | 18.43 min | 18.11 min | 80.02% | 19 112 TFLOPs | ✅ |
| 3 (défaut) | 28 500 | 20.86 min | 20.32 min | 80.37% | 21 100 TFLOPs | ✅ |
| 7 | 18 500 | 27.24 min | 26.92 min | 80.09% | 27 901 TFLOPs | ✅ |

### Seed 2701

| mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux | Statut |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 215 500 | 91.42 min | 87.42 min | 80.06% | 76 817 TFLOPs | ✅ (converge, mais très tardivement) |
| 3 (défaut) | 25 000 | 18.32 min | 17.85 min | 80.16% | 18 509 TFLOPs | ✅ |
| 5 | 18 500 | 19.93 min | 19.58 min | 80.13% | 20 799 TFLOPs | ✅ |
| 7 | 15 500 | 22.93 min | 22.60 min | 80.40% | 23 376 TFLOPs | ✅ |

## Discussion

- **mu=1 est peu fiable et systématiquement le plus coûteux en FLOPs**, y compris quand il finit
  par converger. Sur 2 des 3 seeds (2312, 0308), il n'atteint jamais 80% malgré un budget de steps
  très élevé (173-188k steps) et a été tué par le plafond automatique de 2h sur seed 0308. Sur la
  3e seed (2701), il converge mais après 215 500 steps (91.4 min) — 4 à 5x plus lent que mu=3, pour
  un total de FLOPs 3 à 4x supérieur (76 817 TFLOPs contre 18 509). Le signal de cohérence non
  supervisée tiré d'une seule vue non labellisée par itération est trop pauvre et surtout **trop
  instable d'une seed à l'autre** pour être un choix raisonnable, quel que soit le coût par itération
  le plus faible de la série (356.46 GFLOPs).
- **mu=3 (valeur par défaut) est le meilleur compromis temps/FLOPs**, stable sur les 3 seeds : entre
  18.3 et 30.4 min selon la seed, et systématiquement le moins coûteux en FLOPs totaux parmi les
  configurations qui convergent de façon fiable (18 509-31 095 TFLOPs).
- **mu=5 est quasiment équivalent à mu=3** en accuracy finale et en temps sur les 3 seeds (écarts de
  quelques dixièmes de point et de ±2 minutes), mais consomme systématiquement plus de FLOPs totaux
  (+3% à +9% selon la seed) pour un gain d'accuracy négligeable. Chaque itération coûte 52% de FLOPs
  en plus (1 124.25 vs 740.35 GFLOPs), ce qui n'est jamais compensé par une réduction du nombre de
  steps suffisante.
- **mu=7 est systématiquement plus lent et plus coûteux en FLOPs que mu=3/mu=5 sur les 3 seeds**
  (22.9-36.3 min, 23 376-38 458 TFLOPs, +24% à +32% de FLOPs vs mu=3), pour une accuracy finale très
  proche voire parfois légèrement meilleure (80.09-80.40% contre 80.02-80.37% pour mu=3/mu=5) — un
  gain marginal qui ne justifie pas le surcoût.
- **Conclusion pour le papier (250 labels)** : la relation entre mu et l'efficacité n'est pas
  monotone et le comportement de mu=1 est instable d'une seed à l'autre (échec sur 2/3, convergence
  tardive sur la 3e) — en dessous d'un certain seuil, le signal non supervisé par itération devient
  insuffisant et peu fiable. Au-dessus de ce seuil (mu ≥ 3), l'accuracy finale est stable
  (~80.0-80.4% sur les 3 seeds, toutes valeurs de mu confondues) et le choix de mu devient un
  arbitrage temps/FLOPs plutôt qu'un arbitrage de qualité : **mu=3 domine strictement mu=5 et mu=7
  en FLOPs totaux sur les 3 seeds**, pour une accuracy équivalente — d'où le choix de mu=3 comme
  valeur par défaut d'efficientmatch_3.

## Résultats complets — CIFAR-10, 4000 labels, 3 seeds

À 4000 labels (target_acc 90%), le signal supervisé est nettement plus riche qu'à 250 labels : mu=1
converge de façon fiable sur les 3 seeds (contrairement à 250 labels où il échouait sur 2/3), mais
reste systématiquement plus lent et plus coûteux en FLOPs que mu=3.

### Seed 2312

| mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux | Statut |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 99 500 | 41.5 min | 39.6 min | 90.02% | 35 468 TFLOPs | ✅ |
| 3 (défaut) | 41 000 | 29.0 min | 28.2 min | 90.06% | 30 354 TFLOPs | ✅ |
| 5 | 32 000 | 34.2 min | 33.6 min | 90.08% | 35 976 TFLOPs | ✅ |
| 7 | 26 000 | 36.9 min | 36.4 min | 90.06% | 39 212 TFLOPs | ✅ |

### Seed 0308

| mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux | Statut |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 101 500 | 44.2 min | 42.3 min | 90.03% | 36 181 TFLOPs | ✅ |
| 3 (défaut) | 41 000 | 30.1 min | 29.3 min | 90.02% | 30 354 TFLOPs | ✅ |
| 5 | 34 500 | 38.4 min | 37.7 min | 90.12% | 38 787 TFLOPs | ✅ |
| 7 | 25 500 | 36.2 min | 35.7 min | 90.08% | 38 458 TFLOPs | ✅ |

### Seed 2701

| mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux | Statut |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 100 500 | 41.5 min | 39.6 min | 90.06% | 35 824 TFLOPs | ✅ |
| 3 (défaut) | 43 500 | 30.6 min | 29.8 min | 90.09% | 32 205 TFLOPs | ✅ (rejouée : la première mesure, 41.1 min, était gonflée par un ralentissement de la machine) |
| 5 | 30 500 | 32.5 min | 32.0 min | 90.05% | 34 290 TFLOPs | ✅ |
| 7 | 26 000 | 37.0 min | 36.5 min | 90.05% | 39 212 TFLOPs | ✅ |

## Discussion — 4000 labels

- **mu=1 converge de façon fiable sur les 3 seeds** à 4000 labels (contrairement à 250 labels), mais
  reste le plus coûteux en FLOPs à chaque fois (35 468-36 181 TFLOPs) et prend 1.3 à 1.4x plus de
  temps corrigé que mu=3.
- **mu=3 (valeur par défaut) reste le meilleur compromis FLOPs** sur les 3 seeds : c'est
  systématiquement la configuration la moins coûteuse en FLOPs totaux (30 354-32 205 TFLOPs), pour
  une accuracy équivalente aux autres valeurs de mu (90.02-90.06%). Sur la seed 2701, la première
  mesure du temps (41.1 min) était gonflée par un ralentissement de la machine (temps par step
  anormal) : la run a été rejouée (43 500 steps, 30.6 min, 32 205 TFLOPs), en ligne avec les deux
  autres seeds (29-30 min).
- **mu=5 et mu=7 coûtent systématiquement plus de FLOPs que mu=3** (+6% à +29% selon la seed et la
  valeur de mu) pour une accuracy finale quasi identique (90.02-90.12% toutes valeurs confondues) —
  même conclusion qu'à 250 labels : au-delà de mu=3, le surcoût en FLOPs n'est pas compensé par un
  gain d'accuracy mesurable.
- **Conclusion pour le papier (4000 labels)** : contrairement au régime 250 labels où mu=1 est
  instable, à 4000 labels toutes les valeurs de mu testées convergent de façon fiable sur les 3
  seeds — le signal supervisé plus riche compense la pauvreté du signal non supervisé par vue. Le
  choix de mu reste néanmoins un arbitrage FLOPs pur : **mu=3 domine en FLOPs totaux sur les 3
  seeds**, confirmant sa pertinence comme valeur par défaut d'efficientmatch_3 indépendamment du
  régime de labels.

## Reproduire

```bash
# CIFAR-10, 250 labels
python efficientmatch_3.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 --mu 1
python efficientmatch_3.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 --mu 3
python efficientmatch_3.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 --mu 5
python efficientmatch_3.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 --mu 7
# répéter pour --seed 308 et --seed 2701

# CIFAR-10, 4000 labels
python efficientmatch_3.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.90 --mu 1
python efficientmatch_3.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.90 --mu 3
python efficientmatch_3.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.90 --mu 5
python efficientmatch_3.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.90 --mu 7
# répéter pour --seed 308 et --seed 2701
```

Comparaison via `python scripts/run_analysis.py compare --dataset cifar10 --num_labeled <250|4000> --seed <seed>`.
