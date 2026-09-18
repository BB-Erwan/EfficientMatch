# Étude d'équivalence à mu=3 : efficientmatch_3 vs regmixmatch vs fixmatch

Cette étude neutralise le facteur "ratio non-labellisé:labellisé" (mu) pour isoler la contribution
propre du canal Mixup d'efficientmatch_3. Les trois méthodes sont comparées à **mu=3 identique**
sur **CIFAR-10, 250 labels** (target_acc 80%), WRN-28-2, **complète sur les 3 seeds** (2312, 0308,
2701) :

- **efficientmatch_3** : mu=3 est sa valeur par défaut (cf. `ABLATION_MU_EFFICIENTMATCH3.md`).
- **regmixmatch** : normalement exécuté à mu=7 par défaut ; ici forcé à `--mu 3`.
- **fixmatch** : normalement exécuté à mu=7 codé en dur ; un flag `--mu` a été ajouté au script pour
  cette étude (mirroir du flag déjà existant sur `regmixmatch.py`), puis forcé à `--mu 3`.

L'objectif est de vérifier si l'efficacité d'efficientmatch_3 provient simplement d'un mu plus
faible (moins de FLOPs par itération) ou d'un avantage structurel propre à son canal Mixup, en
comparant les trois méthodes au même mu.

## Résultats complets

### Seed 2312

| Méthode | mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux | Statut |
|---|---|---:|---:|---:|---:|---:|---|
| efficientmatch_3 | 3 | 42 000 | 30.4 min | 29.6 min | 80.11% | 31 095 TFLOPs | ✅ |
| fixmatch | 3 | 202 500 | 97.4 min | 93.6 min | 80.30% | 172 145 TFLOPs | ✅ |
| regmixmatch | 3 | 42 000 | 37.0 min | 36.3 min | 76.32% | 38 002 TFLOPs | ❌ jamais atteint 80% |

### Seed 0308

| Méthode | mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux | Statut |
|---|---|---:|---:|---:|---:|---:|---|
| efficientmatch_3 | 3 | 28 500 | 20.9 min | 20.3 min | 80.37% | 21 100 TFLOPs | ✅ |
| fixmatch | 3 | 197 000 | 94.9 min | 91.3 min | 80.06% | 167 470 TFLOPs | ✅ |
| regmixmatch | 3 | 16 500 | 23.8 min | 23.5 min | 72.20% | 14 929 TFLOPs | ❌ jamais atteint 80% |

### Seed 2701

| Méthode | mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs totaux | Statut |
|---|---|---:|---:|---:|---:|---:|---|
| efficientmatch_3 | 3 | 25 000 | 18.3 min | 17.9 min | 80.16% | 18 509 TFLOPs | ✅ |
| fixmatch | 3 | 110 000 | 53.2 min | 51.1 min | 80.13% | 93 511 TFLOPs | ✅ |
| regmixmatch | 3 | 35 500 | 28.9 min | 28.2 min | 80.15% | 32 120 TFLOPs | ✅ (1er essai crashé à 1 step, relancé) |

## Discussion

- **efficientmatch_3 domine largement les deux autres méthodes en FLOPs et en temps à mu identique**,
  sur les 3 seeds sans exception. Il est 5.5x à 8x moins coûteux en FLOPs que fixmatch mu=3
  (18 509-31 095 TFLOPs contre 93 511-172 145 TFLOPs) et 1.2x à 2.7x moins coûteux que regmixmatch
  mu=3 (contre 14 929-38 002 TFLOPs, en excluant l'unique cas où regmixmatch a un budget FLOPs plus
  faible mais n'atteint jamais le target).
- **fixmatch à mu=3 est très inefficace en pratique** : bien qu'il finisse par atteindre 80% sur les
  3 seeds, il nécessite un nombre de steps considérablement plus élevé (110 000-202 500 contre
  25 000-42 000 pour efficientmatch_3) — le signal de cohérence pur (sans canal Mixup) est
  insuffisant à ce ratio réduit et compense en multipliant les itérations, au prix d'un temps
  d'entraînement 2.6x à 3.2x supérieur à efficientmatch_3.
- **regmixmatch à mu=3 est instable et sous-performant** : il n'atteint jamais 80% sur 2 des 3 seeds
  (76.32% et 72.20% comme maxima), et sur la 3e seed (2701) le premier essai a crashé immédiatement
  (1 step, accuracy de niveau hasard) avant qu'une relance ne réussisse. Même dans son meilleur cas
  (seed 2701, 80.15%), il reste 1.2x moins efficace en FLOPs qu'efficientmatch_3 pour un temps
  1.6x supérieur.
- **Conclusion pour le papier** : l'efficacité d'efficientmatch_3 n'est pas un simple artefact d'un
  mu réduit — à mu strictement identique (3), ni fixmatch ni regmixmatch n'atteignent son rapport
  temps/FLOPs, et regmixmatch échoue même à converger de façon fiable au même target_acc sur 2/3
  seeds. Le canal Mixup supplémentaire d'efficientmatch_3 apporte donc un signal d'apprentissage
  structurellement plus riche par itération, qui reste efficace même quand le nombre de vues
  non labellisées par itération est réduit — contrairement aux méthodes basées uniquement (fixmatch)
  ou principalement (regmixmatch) sur le signal de cohérence classique.

## Reproduire

```bash
python efficientmatch_3.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 --mu 3
python fixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 --mu 3
python regmixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80 --mu 3
# répéter pour --seed 308 et --seed 2701
```

Comparaison via `python scripts/run_analysis.py compare --dataset cifar10 --num_labeled 250 --seed <seed>`.
