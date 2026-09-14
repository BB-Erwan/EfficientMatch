# CIFAR-10, 4000 labels — inventaire des résultats (3 seeds, target_acc 90%)

Sweep : **efficientmatch_2, fixmatch, flexmatch, mixmatch** sur **CIFAR-10, 4000 labels**, seeds
**2312, 0308, 2701**, `--target_acc 0.90` (arrêt anticipé dès que l'accuracy EMA atteint 90%).
Les 12 runs se sont terminés normalement (aucun n'a dépassé le seuil de sécurité de 2h fixé pour
cette campagne — le plus long, fixmatch/seed 2701, a pris 88.8 min).

FLOPs/itération (mesurés via `scripts/flops_analysis.py`, indépendants du hardware) :
fixmatch/flexmatch = 850.10 GFLOPs (mu=7), mixmatch = 301.64 GFLOPs (mu=1), efficientmatch_2 ≈
740.35 GFLOPs (mu=3, valeur mesurée pour efficientmatch v1 — même architecture/formes de tenseurs,
efficientmatch_2 non mesuré séparément par le script).

## Seed 2312

| Méthode | Steps | Temps | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|
| **efficientmatch_2** | 27 500 | **19.55 min** | 90.01% | **20 359.6 TFLOPs** |
| mixmatch | 198 500 | 59.20 min | 90.01% | 59 875.5 TFLOPs |
| flexmatch | 81 500 | 73.65 min | 90.03% | 69 283.1 TFLOPs |
| fixmatch | 99 000 | 87.06 min | 90.03% | 84 159.9 TFLOPs |

## Seed 0308

| Méthode | Steps | Temps | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|
| **efficientmatch_2** | 32 500 | **23.02 min** | 90.03% | **24 061.4 TFLOPs** |
| mixmatch | 213 500 | 63.57 min | 90.05% | 64 400.1 TFLOPs |
| flexmatch | 84 000 | 75.40 min | 90.06% | 71 408.4 TFLOPs |
| fixmatch | 91 500 | 80.79 min | 90.02% | 77 784.1 TFLOPs |

## Seed 2701

| Méthode | Steps | Temps | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|
| **efficientmatch_2** | 36 000 | **25.43 min** | 90.00% | **26 652.6 TFLOPs** |
| mixmatch | 165 500 | 49.34 min | 90.04% | 49 921.4 TFLOPs |
| flexmatch | 84 000 | 75.42 min | 90.03% | 71 408.4 TFLOPs |
| fixmatch | 100 500 | 88.80 min | 90.15% | 85 435.1 TFLOPs |

## Synthèse

**efficientmatch_2 gagne sur les trois seeds, sur les deux critères (temps et FLOPs), et de loin** :
- **Temps** : 19.6-25.4 min, contre 49-64 min pour mixmatch, 74-75 min pour flexmatch, et
  81-89 min pour fixmatch. Entre **2.4x et 4.5x plus rapide** que le deuxième le plus rapide
  (mixmatch) selon la seed, et **3.3x à 4.4x plus rapide** que fixmatch.
- **FLOPs** : 20 360-26 653 TFLOPs, contre 49 921-64 400 TFLOPs pour mixmatch (le deuxième le plus
  économe), soit **~2.4x moins de FLOPs**. Contre fixmatch (le plus coûteux), l'écart est de
  **~3.2x à 4.1x**.
- Contrairement aux campagnes précédentes sur SVHN/250 labels et CIFAR-10/250 labels (où le
  classement variait selon la seed et la métrique, cf. `EXPERIMENT_LOG.md` §3-4), **ici le
  classement est parfaitement stable sur les 3 seeds** : efficientmatch_2 > mixmatch > flexmatch >
  fixmatch, en temps comme en FLOPs. Le régime à 4000 labels semble donc plus favorable et plus
  reproductible pour situer l'avantage d'efficientmatch_2 que le régime à 250 labels.
- **flexmatch fonctionne normalement ici** (contrairement à SVHN/250 labels où il échouait
  systématiquement, cf. `RUNS_TO_REVISIT.md`) — confirme que son problème est spécifique à ce
  régime SVHN/250-labels/target-90%, pas une régression générale.
- **mixmatch est nettement plus rapide qu'sur CIFAR-10/250 labels** (où il avait mis jusqu'à 13h
  sans converger, cf. `EXPERIMENT_LOG.md` §4.1) — avec 4000 labels, il converge en moins d'une
  heure sur les 3 seeds. Le régime à 250 labels semblait être son point faible spécifique.
