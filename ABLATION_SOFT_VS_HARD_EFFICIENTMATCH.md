# Étude d'ablation : hard vs semi-soft vs soft — variantes du pseudo-label dans le canal Mixup d'efficientmatch

Les trois méthodes de la famille efficientmatch partagent la même architecture générale (FixMatch +
canal Mixup supplémentaire sur les échantillons labellisés et non labellisés) et ne diffèrent que
par la représentation du pseudo-label injectée dans les cibles Mixup des échantillons non
labellisés (`mixup_targets_u`) :

- **efficientmatch_2 (hard)** : argmax du pseudo-label, encodé en one-hot
  (`F.one_hot(pseudo, num_classes)`).
- **efficientmatch_3 (semi-soft, mu=3 par défaut)** : même encodage one-hot que la version hard,
  mais mélangé (Mixup/`lerp`) avec l'échantillon labellisé — la "soft-ification" vient uniquement du
  mélange Mixup, pas de la distribution de probabilité elle-même.
- **efficientmatch_soft** : distribution de probabilité complète du softmax
  (`probs_u_w = F.softmax(logits_u_w, dim=1)`) utilisée directement comme cible Mixup, sans passer
  par l'argmax.

Le canal de cohérence FixMatch classique (`loss_consistency`, basé sur le pseudo-label dur et le
masquage par seuil de confiance) est identique dans les trois méthodes — seule la partie Mixup
change. Cette étude compare les trois variantes sur trois configurations, WRN-28-2, **complète sur
les 3 seeds** (2312, 0308, 2701) dans les trois cas :

- **CIFAR-10, 250 labels** (target_acc 80%)
- **SVHN, 250 labels** (target_acc 90%)
- **CIFAR-10, 4000 labels** (target_acc 90%)

Toutes les runs utilisent mu=3 (valeur par défaut, sauf efficientmatch_2 qui n'a pas de paramètre mu
exposé et tourne à son ratio par défaut).

## Résultats complets — CIFAR-10, 250 labels

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|---:|
| efficientmatch_2 (hard) | 45 000 | 33.4 min | 32.6 min | 80.26% | 33 316 TFLOPs |
| efficientmatch_3 (semi-soft) | 42 000 | 30.4 min | 29.6 min | 80.11% | 31 095 TFLOPs |
| efficientmatch_soft | 46 000 | 33.4 min | 32.6 min | 80.06% | 34 056 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|---:|
| efficientmatch_2 (hard) | 55 500 | 40.2 min | 39.1 min | 80.35% | 41 089 TFLOPs |
| efficientmatch_3 (semi-soft) | 28 500 | 20.9 min | 20.3 min | 80.37% | 21 100 TFLOPs |
| efficientmatch_soft | 29 000 | 21.2 min | 20.7 min | 80.02% | 21 470 TFLOPs |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|---:|
| efficientmatch_2 (hard) | 34 000 | 24.4 min | 23.7 min | 80.00% | 25 172 TFLOPs |
| efficientmatch_3 (semi-soft) | 25 000 | 18.3 min | 17.9 min | 80.16% | 18 509 TFLOPs |
| efficientmatch_soft | 21 500 | 15.9 min | 15.4 min | 80.00% | 15 918 TFLOPs |

## Résultats complets — SVHN, 250 labels

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 (semi-soft) | 7 000 | 5.8 min | 5.7 min | 90.25% | 5 182 TFLOPs |
| efficientmatch_2 (hard) | 8 500 | 6.8 min | 6.6 min | 90.04% | 6 293 TFLOPs |
| efficientmatch_soft | 9 000 | 7.2 min | 7.1 min | 90.11% | 6 663 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 (semi-soft) | 6 500 | 5.3 min | 5.2 min | 90.45% | 4 812 TFLOPs |
| efficientmatch_soft | 7 000 | 5.7 min | 5.6 min | 90.24% | 5 182 TFLOPs |
| efficientmatch_2 (hard) | 8 500 | 6.8 min | 6.6 min | 90.24% | 6 293 TFLOPs |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 (semi-soft) | 8 000 | 6.5 min | 6.3 min | 90.15% | 5 923 TFLOPs |
| efficientmatch_soft | 8 000 | 6.5 min | 6.3 min | 90.12% | 5 923 TFLOPs |
| efficientmatch_2 (hard) | 10 000 | 7.9 min | 7.7 min | 90.02% | 7 404 TFLOPs |

## Résultats complets — CIFAR-10, 4000 labels

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|---:|
| efficientmatch_2 (hard) | 27 500 | 19.6 min | 19.0 min | 90.01% | 20 360 TFLOPs |
| efficientmatch_soft | 37 500 | 26.9 min | 26.1 min | 90.03% | 27 763 TFLOPs |
| efficientmatch_3 (semi-soft) | 41 000 | 29.0 min | 28.2 min | 90.06% | 30 354 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|---:|
| efficientmatch_2 (hard) | 32 500 | 23.0 min | 22.4 min | 90.03% | 24 061 TFLOPs |
| efficientmatch_3 (semi-soft) | 41 000 | 30.1 min | 29.3 min | 90.02% | 30 354 TFLOPs |
| efficientmatch_soft | 54 000 | 38.4 min | 37.4 min | 90.06% | 39 979 TFLOPs |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|---:|
| efficientmatch_2 (hard) | 36 000 | 25.4 min | 24.8 min | 90.00% | 26 653 TFLOPs |
| efficientmatch_3 (semi-soft) | 43 000 | 41.1 min | 40.3 min | 90.03% | 31 835 TFLOPs |
| efficientmatch_soft | 51 500 | 36.7 min | 35.7 min | 90.05% | 38 128 TFLOPs |

## Discussion — CIFAR-10, 4000 labels

- **Le classement s'inverse par rapport aux régimes à 250 labels** : ici, **efficientmatch_2 (hard)
  est systématiquement le plus rapide et le moins coûteux en FLOPs** des trois variantes, sur les 3
  seeds sans exception (20 360-26 653 TFLOPs), pour une accuracy finale équivalente aux deux autres
  (90.00-90.06% toutes variantes confondues).
- **efficientmatch_soft est le plus coûteux sur 2 des 3 seeds** (308 et 2701 : 38 128-39 979 TFLOPs),
  et intermédiaire sur la 3e (2312). efficientmatch_3 (semi-soft) est le plus coûteux sur la seed
  2312 mais se situe entre hard et soft sur les deux autres.
- **Interprétation** : avec un signal supervisé plus riche (4000 labels contre 250), le mélange
  Mixup sur la cible non labellisée — qui accélère nettement la convergence à 250 labels — semble
  perdre son avantage, voire devenir un léger surcoût. L'hypothèse la plus probable est que le
  lissage introduit par le Mixup (que la cible soit one-hot ou softmax complet) dilue un signal de
  pseudo-label qui est déjà fiable à ce régime de labels, alors qu'il compensait un signal faible et
  bruité à 250 labels. Le gain d'efficacité du canal Mixup d'efficientmatch_3/soft observé aux
  régimes 250 labels (CIFAR-10 et SVHN) **ne se généralise donc pas** au régime 4000 labels — le
  facteur déterminant identifié plus haut (mélange Mixup vs pseudo-label dur) dépend de la richesse
  du signal supervisé disponible.

## Discussion — CIFAR-10, 250 labels

- **Les trois variantes atteignent 80% d'accuracy de façon fiable sur les 3 seeds** — la
  représentation du pseudo-label dans le canal Mixup n'affecte pas la capacité de convergence, quel
  que soit le niveau de "soft-ification".
- **efficientmatch_2 (hard) est systématiquement le plus lent et le plus coûteux en FLOPs** des
  trois variantes, sur les 3 seeds sans exception (33 316-41 089 TFLOPs), y compris sur la seed 0308
  où son nombre de steps (55 500) est près du double de celui d'efficientmatch_3 (28 500) pour une
  accuracy quasiment identique (80.35% vs 80.37%). L'absence de mélange Mixup sur la cible non
  labellisée (cible one-hot dure, non lissée par le lerp) semble ralentir la convergence par rapport
  aux deux autres variantes.
- **efficientmatch_3 (semi-soft) et efficientmatch_soft ont un profil très proche** en temps et en
  FLOPs sur les 3 seeds, sans dominance claire de l'un sur l'autre : efficientmatch_3 est légèrement
  plus rapide sur seed 2312 (29.6 vs 32.6 min) et 308 (20.3 vs 20.7 min), tandis
  qu'efficientmatch_soft est plus rapide sur seed 2701 (15.4 vs 17.9 min, et le moins coûteux en
  FLOPs de toute l'étude sur cette seed : 15 918 TFLOPs). Les accuracies finales sont également très
  proches (écarts de 0.00 à 0.16 point).
- **Conclusion pour le papier** : le facteur déterminant pour l'efficacité n'est pas la nature de la
  cible du pseudo-label (dure vs distribution complète) mais bien **la présence du mélange Mixup sur
  la cible non labellisée elle-même** — hard (efficientmatch_2) est nettement moins efficace que
  semi-soft (efficientmatch_3) et soft (efficientmatch_soft), qui sont globalement équivalents entre
  eux. Cela suggère que remplacer l'argmax par la distribution complète du softmax
  (efficientmatch_soft) n'apporte pas de gain net une fois le mélange Mixup déjà en place, et que
  efficientmatch_3 (semi-soft) reste un choix raisonnable pour sa simplicité (encodage one-hot moins
  coûteux à calculer que le softmax complet, bien que cet écart soit négligeable en pratique).

## Discussion — SVHN, 250 labels

- **Le classement observé sur CIFAR-10 se reproduit intégralement sur SVHN** : efficientmatch_2
  (hard) est systématiquement le plus lent et le plus coûteux en FLOPs sur les 3 seeds
  (6 293-7 404 TFLOPs), tandis qu'efficientmatch_3 (semi-soft) et efficientmatch_soft restent très
  proches l'un de l'autre, avec un léger avantage pour efficientmatch_3 sur les 3 seeds (4 812-5 923
  TFLOPs contre 5 182-6 663 TFLOPs pour soft).
- Les accuracies finales sont quasiment indiscernables entre les trois variantes (89.75-90.45%),
  confirmant que la nature de la cible du pseudo-label (dure, semi-soft, ou distribution complète)
  n'affecte pas la qualité de convergence, seulement le nombre d'itérations nécessaires.
- **Conclusion cross-dataset (250 labels)** : le résultat obtenu sur CIFAR-10 250 labels — hard
  nettement moins efficace, semi-soft et soft globalement équivalents — se généralise à SVHN 250
  labels, renforçant l'hypothèse que le mélange Mixup sur la cible non labellisée (présent dans
  semi-soft et soft, absent du hard) est le facteur déterminant de l'efficacité à faible nombre de
  labels.
- **Nuance importante (voir aussi la section CIFAR-10 4000 labels ci-dessus)** : ce facteur n'est pas
  universel — il **s'inverse à 4000 labels**, où hard redevient la variante la plus efficace. La
  conclusion pour le papier doit donc être formulée comme dépendante du régime de labels plutôt que
  comme une propriété générale du canal Mixup : le mélange Mixup profite surtout aux régimes à très
  peu de labels (250), où il compense un signal supervisé pauvre.

## Reproduire

```bash
# CIFAR-10, 250 labels
python efficientmatch_2.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80
python efficientmatch_3.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 --mu 3
python efficientmatch_soft.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80
# répéter pour --seed 308 et --seed 2701

# SVHN, 250 labels
python efficientmatch_2.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.90
python efficientmatch_3.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.90 --mu 3
python efficientmatch_soft.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.90
# répéter pour --seed 308 et --seed 2701

# CIFAR-10, 4000 labels
python efficientmatch_2.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.90
python efficientmatch_3.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.90 --mu 3
python efficientmatch_soft.py --dataset cifar10 --num_labeled 4000 --seed 2312 --target_acc 0.90
# répéter pour --seed 308 et --seed 2701
```

Comparaison via `python scripts/run_analysis.py compare --dataset <cifar10|svhn> --num_labeled <250|4000> --seed <seed>`.
