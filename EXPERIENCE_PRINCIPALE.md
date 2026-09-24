# Expérience principale

Sweep de référence retenu pour l'article : **5 méthodes** (efficientmatch_3 avec mu=3 par défaut,
fixmatch, flexmatch, mixmatch, regmixmatch avec mu=7 par défaut) sur **5 configurations** (SVHN 250
labels, CIFAR-10 250 labels, CIFAR-10 4000 labels, CIFAR-100 10000 labels, CIFAR-100 2500 labels)
et **3 seeds** (2312, 0308, 2701). Toute variante d'ablation (mu différent, efficientmatch
v1/flex/2, sequencematch, regmixmatch_mu3, runs interrompus ou hors seeds retenues) est
**volontairement exclue** de ce document — elle reste documentée dans
`SEEDS_2312_308_2701_INVENTORY.md` (historique complet) et `EXPERIMENT_LOG.md`. **Seule exception :**
la variante **efficientmatch_freematch** (efficientmatch_3 + seuillage adaptatif de FreeMatch,
option `--freematch_threshold`), reportée en ligne supplémentaire "(variante)" dans chaque tableau
et résumée dans la section dédiée ci-dessous ; elle ne fait pas partie des 5 méthodes de référence.

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
| efficientmatch_freematch (variante) | 6 000 | 4.9 min | 4.8 min | 90.22% | 4 442 TFLOPs |
| fixmatch | 8 000 | 7.8 min | 7.7 min | 90.01% | 6 801 TFLOPs |
| flexmatch | 115 500 | 117.9 min | 115.8 min | 83.99% (max) | 98 187 TFLOPs — ❌ jamais atteint 90% (tué à 2h) |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| mixmatch | 52 500 | 18.4 min | 17.4 min | 91.06% | 15 836 TFLOPs |
| efficientmatch_3 | 6 500 | 5.3 min | 5.2 min | 90.45% | 4 812 TFLOPs |
| regmixmatch | 4 500 | 8.2 min | 8.2 min | 90.30% | 8 513 TFLOPs |
| efficientmatch_freematch (variante) | 6 000 | 4.9 min | 4.8 min | 90.11% | 4 442 TFLOPs |
| fixmatch | 6 500 | 6.4 min | 6.3 min | 90.07% | 5 526 TFLOPs |
| flexmatch | 92 000 | 94.3 min | 92.6 min | 82.52% (max) | 78 209 TFLOPs — ❌ jamais atteint 90% (tué à 2h) |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| mixmatch | 35 000 | 12.4 min | 11.8 min | 90.82% | 10 557 TFLOPs |
| regmixmatch | 6 000 | 11.3 min | 11.2 min | 90.37% | 11 351 TFLOPs |
| fixmatch | 8 000 | 7.9 min | 7.7 min | 90.18% | 6 801 TFLOPs |
| efficientmatch_3 | 8 000 | 6.5 min | 6.3 min | 90.15% | 5 923 TFLOPs |
| efficientmatch_freematch (variante) | 6 000 | 4.9 min | 4.8 min | 90.03% | 4 442 TFLOPs |
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
| efficientmatch_freematch (variante) | 41 500 | 29.7 min | 29.0 min | 80.05% | 30 725 TFLOPs |
| regmixmatch | 35 000 | 61.2 min | 60.6 min | 80.04% | 66 215 TFLOPs |
| flexmatch | 65 500 | 64.6 min | 63.3 min | 80.03% | 55 682 TFLOPs |
| mixmatch | 827 500 | 778.7 min (13h) | 763.4 min | 78.42% (max) | 249 607 TFLOPs — ❌ jamais atteint 80% |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 | 28 500 | 20.9 min | 20.3 min | 80.37% | 21 100 TFLOPs |
| efficientmatch_freematch (variante) | 21 500 | 15.7 min | 15.3 min | 80.27% | 15 918 TFLOPs |
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
| efficientmatch_freematch (variante) | 19 500 | 14.2 min | 13.9 min | 80.07% | 14 437 TFLOPs |

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
| efficientmatch_freematch (variante) | 25 500 | 18.2 min | 17.7 min | 90.09% | 18 879 TFLOPs |
| efficientmatch_3 | 41 000 | 29.0 min | 28.2 min | 90.06% | 30 354 TFLOPs |
| regmixmatch | 25 000 | 43.4 min | 42.9 min | 90.05% | 47 296 TFLOPs |
| fixmatch | 99 000 | 87.1 min | 85.2 min | 90.03% | 84 160 TFLOPs |
| flexmatch | 60 500 | 59.2 min | 58.1 min | 90.02% | 51 431 TFLOPs |
| mixmatch | 198 500 | 59.2 min | 55.5 min | 90.01% | 59 876 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_freematch (variante) | 28 500 | 20.2 min | 19.7 min | 90.12% | 21 100 TFLOPs |
| efficientmatch_3 | 41 000 | 30.1 min | 29.3 min | 90.02% | 30 354 TFLOPs |
| mixmatch | 213 500 | 63.6 min | 59.6 min | 90.05% | 64 400 TFLOPs |
| regmixmatch | 27 000 | 46.7 min | 46.2 min | 90.02% | 51 080 TFLOPs |
| fixmatch | 91 500 | 80.8 min | 79.1 min | 90.02% | 77 784 TFLOPs |
| flexmatch | 90 000 | 88.1 min | 86.4 min | 90.07% | 76 509 TFLOPs |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_3 (rejoué, temps propre) | 43 500 | 30.6 min | 29.8 min | 90.09% | 32 205 TFLOPs |
| efficientmatch_freematch (variante) | 31 000 | 22.0 min | 21.4 min | 90.05% | 22 951 TFLOPs |
| regmixmatch | 27 000 | 47.7 min | 47.2 min | 90.06% | 51 080 TFLOPs |
| mixmatch | 165 500 | 49.3 min | 46.3 min | 90.04% | 49 921 TFLOPs |
| flexmatch | 83 500 | 78.3 min | 76.8 min | 90.03% | 70 983 TFLOPs |
| fixmatch | 100 500 | 88.8 min | 86.9 min | 90.15% | 85 435 TFLOPs |

**Synthèse** : **efficientmatch_3 est systématiquement le plus rapide et le plus économe en FLOPs**
parmi les 5 méthodes de référence (29.0-30.1 min, 30.4-32.2k TFLOPs), suivi de regmixmatch et
mixmatch en milieu de classement. flexmatch et fixmatch restent les plus lents, sans ordre stable
entre eux selon la seed. La run efficientmatch_3 de la seed 2701 a été rejouée : la première
mesure (41.1 min, 57 ms/step) était gonflée d'environ 30% par la machine, le temps propre est de
30.6 min à 42 ms/step (cf. "Fiabilité des temps" ci-dessous).

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
| regmixmatch (rejoué, temps propre) | 17 500 | 61.9 min | 61.0 min | 50.41% | 130 673 TFLOPs |
| flexmatch | 31 500 | 70.8 min | 69.2 min | 50.25% | 105 679 TFLOPs |
| efficientmatch_freematch (variante) | 16 000 | 29.8 min | 29.0 min | 50.18% | 46 751 TFLOPs |
| efficientmatch_3 | 30 464 | 58.1 min | 55.1 min | 50.11% | 89 014 TFLOPs |
| fixmatch | 54 000 | 119.2 min | 116.5 min | 46.81% (max) | 181 164 TFLOPs — ❌ jamais atteint 50% |
| mixmatch | 84 000 | 67.9 min | 63.7 min | 46.35% (max) | 99 996 TFLOPs — ❌ jamais atteint 50% |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_freematch (variante) | 14 500 | 27.1 min | 26.3 min | 51.04% | 42 368 TFLOPs |
| flexmatch | 38 000 | 84.8 min | 82.8 min | 50.99% | 127 485 TFLOPs |
| regmixmatch | 15 000 | 53.3 min | 52.5 min | 50.50% | 112 005 TFLOPs |
| efficientmatch_3 | 26 500 | 49.5 min | 48.1 min | 50.12% | 77 431 TFLOPs |
| fixmatch | 48 500 | 107.3 min | 104.8 min | 48.13% (max) | 162 712 TFLOPs — ❌ jamais atteint 50% |
| mixmatch | 125 000 | 98.4 min | 92.1 min | 46.48% (max) | 148 804 TFLOPs — ❌ jamais atteint 50% |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_freematch (variante) | 16 000 | 29.7 min | 28.9 min | 50.60% | 46 751 TFLOPs |
| regmixmatch | 20 000 | 71.1 min | 70.1 min | 50.56% | 149 340 TFLOPs |
| flexmatch | 31 500 | 69.6 min | 68.0 min | 50.45% | 105 679 TFLOPs |
| efficientmatch_3 | 32 000 | 59.6 min | 58.0 min | 50.40% | 93 502 TFLOPs |
| fixmatch | 53 000 | 116.0 min | 113.4 min | 47.04% (max) | 177 809 TFLOPs — ❌ jamais atteint 50% |
| mixmatch | 137 500 | 108.0 min | 101.1 min | 45.72% (max) | 163 684 TFLOPs — ❌ jamais atteint 50% (tué à 2h) |

**Synthèse** : **efficientmatch_3 est systématiquement le plus rapide et le moins coûteux en
FLOPs** parmi les méthodes qui convergent (48.1-58.1 min corrigé, 77-93k TFLOPs), devant regmixmatch
et flexmatch (classement variable selon la seed, 52.5-82.8 min ; regmixmatch, rejoué sur la seed
2312 avec un temps propre de 61.0 min, passe derrière efficientmatch_3 au lieu des 85.6 min
gonflés de la première mesure). **fixmatch et mixmatch échouent
systématiquement à atteindre 50%** sur les 3 seeds, plafonnant respectivement autour de 46-48% et
45-47% avant d'être arrêtés par le plafond watchdog de 2h (sur les 3 seeds pour les deux méthodes) —
contrairement à CIFAR-100 10000 labels (target 60%) où fixmatch converge normalement ; le régime à
2500 labels (25/classe sur 100 classes) semble être un point de rupture pour ces deux méthodes sur
cette architecture.

Note de lecture : les colonnes Steps/Temps des runs tuées à 2h indiquent le point où l'accuracy a
été **maximale**, pas le dernier point enregistré (les runs ont bien tourné jusqu'à ~119-120 min ;
l'accuracy de fixmatch et mixmatch oscille en fin de run, leur pic peut donc précéder l'arrêt).

---

## Variante efficientmatch_freematch (seuillage adaptatif FreeMatch)

efficientmatch_3 dont le seuil de confiance fixe (tau = 0.95) est remplacé par le seuillage
adaptatif de FreeMatch tel qu'implémenté dans `regmixmatch.py` : `seuil = time_p × p_model[classe] /
max(p_model)`, avec `time_p` et `p_model` suivis par EMA (0.999) sur les prédictions faibles,
initialisés à 1/nb_classes (init standard de FreeMatch — `regmixmatch.py` utilise à la place un
échauffement supervisé de 2048 pas, non reproduit ici pour garder le budget d'entraînement
inchangé), et borné à [0.9, 0.95] sur SVHN. Le masque adaptatif sert à la fois à la loss de
cohérence et au filtrage du Mixup. Même cible et même plafond 2h que les méthodes de référence.

**FLOPs exacts.** Le seuillage n'ajoute aucun appel de modèle, mais des opérations élémentaires
par itération (moyenne de confiance, deux EMA, max, produit, comparaisons) que `FlopCounterMode`
(conv/matmul uniquement) ne voit pas ; elles sont comptées à la main (1 FLOP par addition,
multiplication, division, comparaison ou max ; indexation et copies = 0 ; le `>=` final est déjà
présent avec un tau fixe, donc pas un surcoût) dans `run_analysis.freematch_threshold_flops`, et
ajoutées au coût mesuré du modèle dans `flops_analysis.py` :

| Architecture / config | efficientmatch_3 | efficientmatch_freematch | Surcoût du seuillage |
|---|---:|---:|---:|
| WRN-28-2, CIFAR-10 (10 classes) | 740 351 508 480 FLOPs/it | 740 351 510 836 | +2 356 (3.2e-9 relatif) |
| WRN-28-2, SVHN (10 classes, borne active) | 740 351 508 480 | 740 351 511 220 | +2 740 (3.7e-9) |
| WRN-28-4, CIFAR-100 (100 classes) | 2 921 930 883 072 | 2 921 930 903 158 | +20 086 (6.9e-9) |

Sur une run complète le seuillage représente au plus 3.2e8 FLOPs (CIFAR-100 2500), soit ~3e-4 TFLOPs :
les TFLOPs des tableaux ci-dessus (arrondis à l'unité) sont donc identiques à ceux d'un calcul
sans le surcoût — c'est le résultat exact, pas une approximation.

**Gain vs efficientmatch_3, 3 seeds** (steps = métrique indépendante du matériel ; le temps corrigé
donne des écarts quasi identiques) :

| Config | Seed 2312 | Seed 0308 | Seed 2701 | Moyenne |
|---|---:|---:|---:|---:|
| SVHN, 250 labels (borne active) | -14% | -8% | -25% | -16% |
| CIFAR-10, 250 labels | -1% | -25% | -22% | -16% |
| CIFAR-10, 4000 labels | -38% | -31% | -29% | -32% |
| CIFAR-100, 2500 labels (WF4) | -47% | -45% | -50% | -48% |

La variante est la plus rapide et la moins coûteuse en FLOPs de toutes les méthodes sur les 12
combinaisons config × seed. Le gain croît avec le nombre de labels et de classes (~-16% sur SVHN
et CIFAR-10 250, -32% sur CIFAR-10 4000, -48% sur CIFAR-100 2500) ; il est très variable d'une
seed à l'autre sur CIFAR-10 250 (de -1% à -25%).

**La borne [0.9, 0.95] sur SVHN est indispensable.** Elle vient du code de référence (FreeMatch /
RegMixMatch : `freematch_utils.consistency_loss`), que `regmixmatch.py` reproduit et que la variante
reprend. Sans elle (`--freematch_svhn_clamp false`, résultats `efficientmatch_freematch_noclamp_*`),
la variante n'atteint jamais 90% sur les 3 seeds : maximum de 85.5% (seed 2312, arrêtée à la main
après 51 min), 84.2% (0308, coupée à 25 min) et 86.3% (2701, coupée à 25 min), puis lente
dégradation vers 82-83% (masque ~84% des échantillons, qualité des pseudo-labels ~0.81 contre 0.88
avec la borne à step égal). Le seuil adaptatif seul (moyenne de confiance suivie par EMA) reste
trop bas sur SVHN : trop de pseudo-labels faux entrent dans le masque et le modèle se renforce sur
ses erreurs. Même comportement pour regmixmatch sans borne (`--svhn_clamp false`, seed 2312,
2h) : plafond à 84.65% au step 19 000 (33.6 min), 81.8% à 70 500 steps (119 min), contre 90.51% en
5 500 steps / 10.3 min avec la borne. Ces runs sont des diagnostics (budget de 25 min à 2h, hors
plafond 2h uniforme) et ne font pas partie des tableaux de référence.

---

## Fiabilité des temps (runs rejouées)

Les steps et les FLOPs sont déterministes ; le **temps d'horloge** ne l'est pas. Un contrôle du temps
par step sur toutes les runs a montré que la machine a parfois tourné 20 à 200% plus lentement
qu'à l'habitude, sans autre run concurrente enregistrée (cause non identifiée : charge externe ou
état du GPU). Runs de l'expérience principale concernées (temps par step >20% au-dessus de la
médiane de la même méthode et config) :

| Run | ms/step | Médiane | Traitement |
|---|---:|---:|---|
| efficientmatch_3, CIFAR-10 4000, seed 2701 | 57.4 | 44 | **rejouée** : 30.6 min à 42 ms/step (au lieu de 41.1 min) |
| regmixmatch, CIFAR-100 2500, seed 2312 | 273 | 213 | **rejouée** : 61.9 min à 212 ms/step (au lieu de 87.5 min) |
| mixmatch, CIFAR-10 250, seed 2312 | 56.5 | 18.8 | **non rejouée** (plafond 2h impossible à tenir) : temps estimé à vitesse nominale ≈ 260 min au lieu de 779 min ; steps (827 500) et accuracy inchangés |
| efficientmatch_3, CIFAR-100 10000, seed 2701 | 156 | 126 | non rejouée : temps estimé ≈ 20 min au lieu de 25.3 min |

Les deux reruns ont remplacé les anciens résultats (récupérables via l'historique git).

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
dans le périmètre retenu pour l'article. La variante efficientmatch_freematch est également
complète (3 seeds) sur SVHN 250, CIFAR-10 250, CIFAR-10 4000 et CIFAR-100 2500 (pas sur
CIFAR-100 10000).
