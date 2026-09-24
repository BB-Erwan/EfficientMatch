# Expérience principale

Sweep de référence retenu pour l'article : **5 méthodes** (efficientmatch_3 avec mu=3 par défaut,
fixmatch, flexmatch, mixmatch, regmixmatch avec mu=7 par défaut) sur **5 configurations** (SVHN 250
labels, CIFAR-10 250 labels, CIFAR-10 4000 labels, CIFAR-100 10000 labels, CIFAR-100 2500 labels)
et **3 seeds** (2312, 0308, 2701). Toute variante d'ablation (mu différent, efficientmatch
v1/flex/2, sequencematch, regmixmatch_mu3, runs interrompus ou hors seeds retenues) est
**volontairement exclue** de ce document — elle reste documentée dans
`SEEDS_2312_308_2701_INVENTORY.md` (historique complet) et `EXPERIMENT_LOG.md`.

**CIFAR-100 (10000 et 2500 labels) est reporté exclusivement en WRN-28-4 (WF4)**, l'architecture de
facto sur ces configs (WRN-28-8 sature la VRAM de la carte utilisée, cf. `FLOPS_RESULTS.md`) ; les
anciennes mesures WRN-28-8 ne figurent pas ici. Les 3 autres configs sont en WRN-28-2 (architecture
par défaut).

**Cibles (target_acc)** : SVHN 250 → 90%, CIFAR-10 250 → 80%, CIFAR-10 4000 → 90%, CIFAR-100 10000 →
60%, CIFAR-100 2500 → 50%. Puisque toutes les méthodes d'une même config partagent la même cible,
**l'accuracy finale n'est pas un critère de comparaison pertinent** (c'est le critère d'arrêt) — la
métrique qui compte est le **temps** (et les FLOPs).

**Temps corrigé** = temps brut − (nombre d'évaluations × coût moyen d'une évaluation), coût mesuré
en conditions réelles (cf. `FLOPS_RESULTS.md` §"Coût d'une évaluation") : 553.7 ms pour WRN-28-2,
1 499.1 ms pour WRN-28-4. C'est l'estimation à préférer pour comparer la vitesse de convergence
intrinsèque des méthodes ; le "Temps" brut reste la mesure de référence du temps de run observé.

Runs soumis à un plafond automatique de 2h (watchdog) : au-delà, le run est tué et la valeur
rapportée est le maximum atteint avant coupure (marqué "jamais atteint la cible").

---

## SVHN, 250 labels (target 90%)

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| regmixmatch | 5 500 | 10.3 min | 10.2 min | 90.51% | 10 405 TFLOPs |
| mixmatch | 18 500 | 6.8 min | 6.5 min | 90.27% | 5 580 TFLOPs |
| efficientmatch_3 | 7 000 | 5.8 min | 5.7 min | 90.25% | 5 182 TFLOPs |
| fixmatch | 8 000 | 7.8 min | 7.7 min | 90.01% | 6 801 TFLOPs |
| flexmatch | 115 500 | 117.9 min | 115.8 min | 83.99% (max) | 98 187 TFLOPs — ❌ jamais atteint 90% (tué à 2h) |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| mixmatch | 52 500 | 18.4 min | 17.4 min | 91.06% | 15 836 TFLOPs |
| efficientmatch_3 | 6 500 | 5.3 min | 5.2 min | 90.45% | 4 812 TFLOPs |
| regmixmatch | 4 500 | 8.2 min | 8.2 min | 90.30% | 8 513 TFLOPs |
| fixmatch | 6 500 | 6.4 min | 6.3 min | 90.07% | 5 526 TFLOPs |
| flexmatch | 92 000 | 94.3 min | 92.6 min | 82.52% (max) | 78 209 TFLOPs — ❌ jamais atteint 90% (tué à 2h) |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| mixmatch | 35 000 | 12.4 min | 11.8 min | 90.82% | 10 557 TFLOPs |
| regmixmatch | 6 000 | 11.3 min | 11.2 min | 90.37% | 11 351 TFLOPs |
| fixmatch | 8 000 | 7.9 min | 7.7 min | 90.18% | 6 801 TFLOPs |
| efficientmatch_3 | 8 000 | 6.5 min | 6.3 min | 90.15% | 5 923 TFLOPs |
| flexmatch | 57 500 | 59.4 min | 58.4 min | 86.73% (max) | 48 881 TFLOPs — ❌ jamais atteint 90% (tué à 2h) |

**Synthèse** : **flexmatch échoue sur les 3 seeds** (plafond 83-87%, jamais 90%, tué par le watchdog
2h à chaque fois) — problème méthodologique reproductible, pas un artefact de bug (le fix
`thresh_warmup` déjà en place a été vérifié conforme à la référence officielle
`microsoft/Semi-supervised-learning`). Parmi les 4 méthodes qui convergent, le classement varie
d'une seed à l'autre entre regmixmatch, mixmatch, efficientmatch_3 et fixmatch, mais toutes restent
dans une fourchette étroite (5.2-18.4 min).

---

## CIFAR-10, 250 labels (target 80%)

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 | 42 000 | 30.4 min | 29.6 min | 80.11% | 31 095 TFLOPs |
| fixmatch | 170 500 | 272.5 min | 269.3 min | 80.06% | 144 942 TFLOPs |
| fixmatch (bis, même config) | 127 000 | 116.0 min | 113.7 min | 80.05% | 107 963 TFLOPs |
| regmixmatch | 35 000 | 61.2 min | 60.6 min | 80.04% | 66 215 TFLOPs |
| flexmatch | 65 500 | 64.6 min | 63.3 min | 80.03% | 55 682 TFLOPs |
| mixmatch | 827 500 | 778.7 min (13h) | 763.4 min | 78.42% (max) | 249 607 TFLOPs — ❌ jamais atteint 80% |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 | 28 500 | 20.9 min | 20.3 min | 80.37% | 21 100 TFLOPs |
| flexmatch | 52 500 | 52.3 min | 51.3 min | 80.16% | 44 630 TFLOPs |
| mixmatch | 883 000 | 276.6 min (4h37) | 260.3 min | 80.07% | 266 348 TFLOPs |
| regmixmatch | 25 500 | 44.3 min | 43.8 min | 80.02% | 48 242 TFLOPs |
| fixmatch | 64 000 | 59.4 min | 58.2 min | 80.00% | 54 406 TFLOPs |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 | 25 000 | 18.3 min | 17.9 min | 80.16% | 18 509 TFLOPs |
| regmixmatch | 21 000 | 36.5 min | 36.1 min | 80.20% | 39 729 TFLOPs |
| flexmatch | 42 000 | 41.5 min | 40.7 min | 80.17% | 35 704 TFLOPs |
| fixmatch | 61 500 | 57.4 min | 56.3 min | 80.15% | 52 281 TFLOPs |
| mixmatch | 464 500 | 145.6 min (2h26) | 137.0 min | 80.10% | 140 112 TFLOPs |

**Synthèse** : **efficientmatch_3 est systématiquement le plus rapide** (18.3-30.4 min), suivi de
flexmatch et regmixmatch (classement variable selon la seed). **mixmatch est nettement le point
faible de cette config** : catastrophique sur seed 2312 (n'atteint jamais 80% en 13h), et très lent
même quand il converge sur les 2 autres seeds (137-260 min). fixmatch reste globalement lent
(57-272 min), avec un cas extrême sur seed 2312 (272.5 min) — **un second run (bis) sur cette même
config a mis 116.0 min, soit plus de 2x plus rapide avec un code strictement identique**, signe
d'une forte variance run-à-run pour fixmatch sur cette config plutôt que d'un problème
méthodologique systématique.

---

## CIFAR-10, 4000 labels (target 90%)

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 | 41 000 | 29.0 min | 28.2 min | 90.06% | 30 354 TFLOPs |
| regmixmatch | 25 000 | 43.4 min | 42.9 min | 90.05% | 47 296 TFLOPs |
| fixmatch | 99 000 | 87.1 min | 85.2 min | 90.03% | 84 160 TFLOPs |
| flexmatch | 60 500 | 59.2 min | 58.1 min | 90.02% | 51 431 TFLOPs |
| mixmatch | 198 500 | 59.2 min | 55.5 min | 90.01% | 59 876 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 | 41 000 | 30.1 min | 29.3 min | 90.02% | 30 354 TFLOPs |
| mixmatch | 213 500 | 63.6 min | 59.6 min | 90.05% | 64 400 TFLOPs |
| regmixmatch | 27 000 | 46.7 min | 46.2 min | 90.02% | 51 080 TFLOPs |
| fixmatch | 91 500 | 80.8 min | 79.1 min | 90.02% | 77 784 TFLOPs |
| flexmatch | 90 000 | 88.1 min | 86.4 min | 90.07% | 76 509 TFLOPs |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 | 43 000 | 41.1 min | 40.3 min | 90.03% | 31 835 TFLOPs |
| regmixmatch | 27 000 | 47.7 min | 47.2 min | 90.06% | 51 080 TFLOPs |
| mixmatch | 165 500 | 49.3 min | 46.3 min | 90.04% | 49 921 TFLOPs |
| flexmatch | 83 500 | 78.3 min | 76.8 min | 90.03% | 70 983 TFLOPs |
| fixmatch | 100 500 | 88.8 min | 86.9 min | 90.15% | 85 435 TFLOPs |

**Synthèse** : **efficientmatch_3 est systématiquement le plus rapide et le plus économe en FLOPs**
(29.0-41.1 min, 30.4-31.8k TFLOPs), suivi de regmixmatch et mixmatch en milieu de classement.
flexmatch et fixmatch restent les plus lents, sans ordre stable entre eux selon la seed.

---

## CIFAR-100, 10000 labels (target 60%, WRN-28-4)

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| mixmatch | 12 800 | 11.3 min | 10.0 min | 60.01% | 15 238 TFLOPs |
| efficientmatch_3 | 10 240 | 19.8 min | 18.8 min | 60.00% | 29 921 TFLOPs |
| regmixmatch | 5 376 | 25.1 min | 24.6 min | 60.07% | 40 143 TFLOPs |
| fixmatch | 12 544 | 29.0 min | 27.8 min | 60.00% | 42 084 TFLOPs |
| flexmatch | 13 568 | 33.7 min | 32.3 min | 60.06% | 45 519 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| mixmatch | 12 800 | 11.3 min | 10.0 min | 60.08% | 15 238 TFLOPs |
| efficientmatch_3 | 10 240 | 21.4 min | 20.4 min | 60.29% | 29 921 TFLOPs |
| regmixmatch | 5 888 | 28.0 min | 27.4 min | 60.31% | 43 966 TFLOPs |
| fixmatch | 12 800 | 29.2 min | 28.0 min | 60.01% | 42 942 TFLOPs |
| flexmatch | 12 288 | 30.5 min | 29.2 min | 60.51% | 41 225 TFLOPs |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| mixmatch | 13 568 | 11.9 min | 10.6 min | 60.11% | 16 152 TFLOPs |
| regmixmatch | 6 144 | 25.2 min | 24.6 min | 60.14% | 45 877 TFLOPs |
| efficientmatch_3 | 9 728 | 25.3 min | 24.3 min | 60.05% | 28 425 TFLOPs |
| flexmatch | 11 264 | 26.0 min | 24.9 min | 60.20% | 37 789 TFLOPs |
| fixmatch | 13 568 | 31.1 min | 29.7 min | 60.00% | 45 519 TFLOPs |

**Synthèse** : **mixmatch est systématiquement le plus rapide et le plus économe en FLOPs** sur
cette config (10.0-20.4 min corrigé selon la seed, 15.2-16.2k TFLOPs quand il domine), suivi
d'efficientmatch_3 (18.8-24.3 min). regmixmatch et fixmatch se tiennent en milieu de classement
(~24-28 min), flexmatch est le plus lent des méthodes qui convergent (25-33 min). **Couverture
désormais complète sur les 3 seeds pour les 5 méthodes.**

---

## CIFAR-100, 2500 labels (target 50%, WRN-28-4)

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| regmixmatch | 19 200 | 87.5 min | 85.6 min | 51.41% | 143 367 TFLOPs |
| flexmatch | 31 500 | 70.8 min | 69.2 min | 50.25% | 105 679 TFLOPs |
| efficientmatch_3 | 30 464 | 58.1 min | 55.1 min | 50.11% | 89 014 TFLOPs |
| fixmatch | 54 000 | 119.2 min | 116.5 min | 46.81% (max) | 181 164 TFLOPs — ❌ jamais atteint 50% |
| mixmatch | 84 000 | 67.9 min | 63.7 min | 46.35% (max) | 99 996 TFLOPs — ❌ jamais atteint 50% |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| flexmatch | 38 000 | 84.8 min | 82.8 min | 50.99% | 127 485 TFLOPs |
| regmixmatch | 15 000 | 53.3 min | 52.5 min | 50.50% | 112 005 TFLOPs |
| efficientmatch_3 | 26 500 | 49.5 min | 48.1 min | 50.12% | 77 431 TFLOPs |
| fixmatch | 48 500 | 107.3 min | 104.8 min | 48.13% (max) | 162 712 TFLOPs — ❌ jamais atteint 50% |
| mixmatch | 125 000 | 98.4 min | 92.1 min | 46.48% (max) | 148 804 TFLOPs — ❌ jamais atteint 50% |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| regmixmatch | 20 000 | 71.1 min | 70.1 min | 50.56% | 149 340 TFLOPs |
| flexmatch | 31 500 | 69.6 min | 68.0 min | 50.45% | 105 679 TFLOPs |
| efficientmatch_3 | 32 000 | 59.6 min | 58.0 min | 50.40% | 93 502 TFLOPs |
| fixmatch | 53 000 | 116.0 min | 113.4 min | 47.04% (max) | 177 809 TFLOPs — ❌ jamais atteint 50% |
| mixmatch | 137 500 | 108.0 min | 101.1 min | 45.72% (max) | 163 684 TFLOPs — ❌ jamais atteint 50% (tué à 2h) |

**Synthèse** : **efficientmatch_3 est systématiquement le plus rapide et le moins coûteux en
FLOPs** parmi les méthodes qui convergent (48.1-58.1 min corrigé, 77-93k TFLOPs), devant regmixmatch
et flexmatch (classement variable selon la seed, 48.1-85.6 min). **fixmatch et mixmatch échouent
systématiquement à atteindre 50%** sur les 3 seeds, plafonnant respectivement autour de 46-48% et
45-47% avant d'être arrêtés par le plafond watchdog de 2h (sur les 3 seeds pour les deux méthodes) —
contrairement à CIFAR-100 10000 labels (target 60%) où fixmatch converge normalement ; le régime à
2500 labels (25/classe sur 100 classes) semble être un point de rupture pour ces deux méthodes sur
cette architecture.

Note de lecture : les colonnes Steps/Temps des runs tuées à 2h indiquent le point où l'accuracy a
été **maximale**, pas le dernier point enregistré (les runs ont bien tourné jusqu'à ~119-120 min ;
l'accuracy de fixmatch et mixmatch oscille en fin de run, leur pic peut donc précéder l'arrêt).

---

## Vue d'ensemble

| Config | efficientmatch_3 | fixmatch | flexmatch | mixmatch | regmixmatch |
|---|:---:|:---:|:---:|:---:|:---:|
| SVHN, 250 labels | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 (**0/3 atteint 90%**) | ✅ 3/3 | ✅ 3/3 |
| CIFAR-10, 250 labels | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 (1 jamais convergé) | ✅ 3/3 |
| CIFAR-10, 4000 labels | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 |
| CIFAR-100, 10000 labels (WF4) | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 |
| CIFAR-100, 2500 labels (WF4) | ✅ 3/3 | ✅ 3/3 (**0/3 atteint 50%**) | ✅ 3/3 | ✅ 3/3 (**0/3 atteint 50%**) | ✅ 3/3 |

**Couverture complète sur les 3 seeds pour les 5 configs et les 5 méthodes.** Aucun trou restant
dans le périmètre retenu pour l'article.
