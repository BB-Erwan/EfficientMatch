# CIFAR-10, 4000 labels — inventaire des résultats (3 seeds, target_acc 90%)

Sweep : **efficientmatch_2, efficientmatch_3, fixmatch, flexmatch, mixmatch, regmixmatch,
sequencematch** sur **CIFAR-10, 4000 labels**, seeds **2312, 0308, 2701**, `--target_acc 0.90`
(arrêt anticipé dès que l'accuracy EMA atteint 90%), architecture WRN-28-2 (widen_factor par
défaut sur ce dataset). Les temps sont corrigés du coût d'évaluation (`corrected`, via
`run_analysis.py compare`) quand mentionné ; sinon il s'agit du temps brut (`@min`).

**Comme toutes les méthodes partagent le même `target_acc`, l'accuracy finale n'est pas un
critère de comparaison pertinent (c'est le critère d'arrêt) — la métrique qui compte est le
temps.**

FLOPs/itération (mesurés via `scripts/flops_analysis.py`, WRN-28-2, indépendants du hardware) :
fixmatch/flexmatch = 850.10 GFLOPs (mu=7), mixmatch = 301.64 GFLOPs (mu=1), efficientmatch_2 ≈
740.35 GFLOPs (mu=3), efficientmatch_3 = 740.35 GFLOPs (mu=3), sequencematch = 1809.61 GFLOPs
(mu=7), regmixmatch = 1891.86 GFLOPs (mu=7).

## Seed 2312

| Méthode | Steps | Temps | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|
| **efficientmatch_2** | 27 500 | **19.6 min** | 90.01% | **20 360 TFLOPs** |
| efficientmatch_3 | 41 000 | 29.0 min | 90.06% | 30 354 TFLOPs |
| sequencematch | 23 000 | 40.2 min | 90.05% | 41 621 TFLOPs |
| regmixmatch | 25 000 | 43.4 min | 90.05% | 47 296 TFLOPs |
| mixmatch | 198 500 | 59.2 min | 90.01% | 59 876 TFLOPs |
| flexmatch | 81 500 | 73.7 min | 90.03% | 69 283 TFLOPs |
| fixmatch | 99 000 | 87.1 min | 90.03% | 84 160 TFLOPs |

*(Run d'ablation isolé sur cette seed uniquement, hors sweep principal : `efficientmatch_flex`
atteint 90.97% en 41.0 min / 42 200 TFLOPs — non comparable directement, pas rejoué sur les
2 autres seeds.)*

## Seed 0308

| Méthode | Steps | Temps | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|
| **efficientmatch_2** | 32 500 | **23.0 min** | 90.03% | **24 061 TFLOPs** |
| efficientmatch_3 | 41 000 | 30.1 min | 90.02% | 30 354 TFLOPs |
| sequencematch | 21 500 | 38.9 min | 90.01% | 38 907 TFLOPs |
| regmixmatch | 27 000 | 46.7 min | 90.02% | 51 080 TFLOPs |
| mixmatch | 213 500 | 63.6 min | 90.05% | 64 400 TFLOPs |
| flexmatch | 84 000 | 75.4 min | 90.06% | 71 408 TFLOPs |
| fixmatch | 91 500 | 80.8 min | 90.02% | 77 784 TFLOPs |

## Seed 2701

| Méthode | Steps | Temps | Acc finale | FLOPs totaux |
|---|---:|---:|---:|---:|
| **efficientmatch_2** | 36 000 | **25.4 min** | 90.00% | **26 653 TFLOPs** |
| efficientmatch_3 | — | *en attente* | — | — |
| sequencematch | — | *en attente* | — | — |
| mixmatch | 165 500 | 49.3 min | 90.04% | 49 921 TFLOPs |
| flexmatch | 84 000 | 75.4 min | 90.03% | 71 408 TFLOPs |
| regmixmatch | 27 000 | 47.7 min | 90.06% | 51 080 TFLOPs |
| fixmatch | 100 500 | 88.8 min | 90.15% | 85 435 TFLOPs |

## Synthèse

**efficientmatch_2 reste le plus rapide sur les trois seeds**, mais **efficientmatch_3 est
systématiquement le deuxième plus rapide** sur les deux seeds testées à ce jour :
- **Temps** : efficientmatch_2 à 19.6-25.4 min ; efficientmatch_3 juste derrière à 29.0-30.1 min
  (~1.3-1.5x plus lent qu'efficientmatch_2, mais 1.3-1.4x plus rapide que sequencematch, le
  troisième). sequencematch et regmixmatch se disputent la 3e/4e place (38.9-46.7 min). mixmatch,
  flexmatch et fixmatch restent nettement plus lents (49-89 min).
- **FLOPs** : efficientmatch_3 partage le même coût par itération qu'efficientmatch_2 (740.35
  GFLOPs, mu=3) mais nécessite davantage de steps pour converger, d'où un total FLOPs ~1.3-1.5x
  supérieur.
- Classement stable sur les 2 seeds disponibles pour efficientmatch_3 :
  efficientmatch_2 > efficientmatch_3 > sequencematch > regmixmatch > mixmatch > flexmatch >
  fixmatch (ordre approximatif, sequencematch/regmixmatch et mixmatch/flexmatch parfois proches).
- efficientmatch_3/seed 2701 et sequencematch/seed 2701 restent à lancer pour compléter le
  tableau (cf. sweep de comblement en cours, `HANDOFF.md`).
