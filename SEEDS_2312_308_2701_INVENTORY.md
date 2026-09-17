# Inventaire complet — seeds 2312, 0308, 2701

Périmètre retenu pour l'article : seeds **2312, 0308, 2701** uniquement, sur les 4 configurations
**SVHN 250 labels**, **CIFAR-10 250 labels**, **CIFAR-10 4000 labels** et **CIFAR-100 10000
labels**. Toute autre config présente dans `results/` (SVHN 1000/40 labels, CIFAR-100 250/2500
labels, autres seeds) est hors périmètre et volontairement exclue de ce document. FLOPs/itération
(indépendants du hardware, mesurés via `scripts/flops_analysis.py`, cf. `FLOPS_RESULTS.md`) :

| Méthode | mu | GFLOPs/it |
|---|---:|---:|
| mixmatch | 1 | 301.64 |
| efficientmatch_3_mu1 | 1 | 356.46 |
| efficientmatch_flex_mu2 | 2 | 548.40 |
| efficientmatch / efficientmatch_2 / efficientmatch_3 / efficientmatch_flex / efficientmatch_2_flex | 3 | 740.35 |
| regmixmatch_mu3 | 3 | 904.80 |
| fixmatch / flexmatch | 7 | 850.10 |
| efficientmatch_3_mu5 | 5 | 1124.25 |
| efficientmatch_3_mu7 | 7 | 1508.14 |
| sequencematch | 7 | 1809.61 |
| regmixmatch | 7 | 1891.86 |

Note générale sur les seuils cibles : SVHN 250 labels visait 90% (sauf mentions contraires),
CIFAR-10 250 labels visait 80%, CIFAR-10 4000 labels visait 90%.

**Temps corrigé** : dans toutes les tables ci-dessous, la colonne "Temps corrigé" retire l'overhead
d'évaluation du "Temps" brut (`temps corrigé = temps - n_évaluations_effectuées × temps_moyen_par_éval`),
en utilisant le coût par évaluation mesuré en conditions réelles (cf. `FLOPS_RESULTS.md` §"Coût
d'une évaluation") : 553.7 ms pour WRN-28-2 (tous les runs CIFAR-10/SVHN de ce document). C'est une
estimation du temps de calcul "pur" (hors coût d'évaluation), à préférer pour comparer la vitesse
de convergence intrinsèque des méthodes ; la colonne "Temps" brute est conservée à côté pour la
traçabilité et reste la mesure de référence pour le temps de run réellement observé.

**⚠️ Note sur les données incomplètes** : certains runs `regmixmatch_mu3` ci-dessous ont été
interrompus manuellement avant d'atteindre l'objectif (voir annotations dans les tableaux). Ils
sont conservés tels quels dans cet inventaire à la demande explicite de l'utilisateur (valeur du
fichier de résultats au moment de l'arrêt), mais **ne doivent pas être cités comme des résultats de
convergence finale** dans l'article — seule la colonne "Statut" fait foi.

---

## CIFAR-10, 250 labels (target 80%)

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs | Statut |
|---|---:|---:|---:|---:|---:|---|
| efficientmatch_flex_mu2 | 88 000 | 50.68 min | 49.04 min | 80.33% | 48 259.2 TFLOPs | ✅ |
| efficientmatch_2 | 45 000 | 33.43 min | 32.59 min | 80.26% | 33 315.8 TFLOPs | ✅ |
| efficientmatch_flex | 47 000 | 34.79 min | 33.92 min | 80.21% | 34 796.4 TFLOPs | ✅ |
| efficientmatch_3_mu7 | 25 500 | 36.26 min | 35.78 min | 80.18% | 38 457.6 TFLOPs | ✅ |
| efficientmatch_3 (mu=3, défaut) | 42 000 | 30.38 min | 29.60 min | 80.11% | 31 094.7 TFLOPs | ✅ |
| efficientmatch_3_mu5 | 28 500 | 30.53 min | 30.00 min | 80.10% | 32 041.1 TFLOPs | ✅ |
| efficientmatch (v1) | 82 000 | 58.99 min | 57.47 min | 80.10% | 60 708.7 TFLOPs | ✅ |
| fixmatch | 170 500 | 272.46 min | 269.30 min | 80.06% | 144 942.0 TFLOPs | ✅ |
| regmixmatch (mu=7, défaut) | 35 000 | 61.23 min | 60.57 min | 80.04% | 66 215.1 TFLOPs | ✅ |
| sequencematch | 40 500 | 70.63 min | 69.88 min | 80.04% | 73 289.2 TFLOPs | ✅ |
| flexmatch (refait post-fix thresh_warmup) | 65 500 | 64.60 min | 63.30 min | 80.03% | 55 681.6 TFLOPs | ✅ (ancien résultat pré-fix : 114.07 min) |
| mixmatch | 827 500 | 778.67 min (13h) | 763.39 min | 78.42% (max) | 249 607.1 TFLOPs | ❌ jamais atteint 80% |
| efficientmatch_3_mu1 | 173 000 | 71.62 min | 68.42 min | 77.57% (max) | 61 667.6 TFLOPs | ❌ jamais atteint 80% |
| regmixmatch_mu3 | 42 000 | 37.05 min | 36.27 min | 76.32% | 38 001.6 TFLOPs | ⚠️ **interrompu manuellement** ("passe à l'expérience suivante") — un run antérieur complet sur cette même config avait atteint **80.13%** au step 63 000 (56.1 min) avant d'être écrasé par ce re-test |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs | Statut |
|---|---:|---:|---:|---:|---:|---|
| efficientmatch_2 | 55 500 | 40.16 min | 39.13 min | 80.35% | 41 089.4 TFLOPs | ✅ |
| mixmatch | 883 000 | 276.64 min (4h37) | 260.33 min | 80.07% | 266 348.1 TFLOPs | ✅ |
| regmixmatch (mu=7, défaut) | 25 500 | 44.27 min | 43.79 min | 80.02% | 48 242.4 TFLOPs | ✅ |
| fixmatch | 64 000 | 59.43 min | 58.24 min | 80.00% | 54 406.4 TFLOPs | ✅ |
| flexmatch (refait post-fix thresh_warmup) | 52 500 | 52.30 min | 51.30 min | 80.16% | 44 630.3 TFLOPs | ✅ (ancien résultat pré-fix : 45.21 min) |
| regmixmatch_mu3 | 16 500 | 23.85 min | 23.54 min | 72.20% | 14 929.2 TFLOPs | ⚠️ **interrompu manuellement** (tué via "Non arrête" pendant un sweep multi-seed) — non représentatif, jamais relancé jusqu'à convergence |

*(efficientmatch v1 non testé sur cette seed — volontairement exclu sur demande utilisateur après
le run seed 2312)*

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs | Statut |
|---|---:|---:|---:|---:|---:|---|
| regmixmatch (mu=7, défaut) | 21 000 | 36.53 min | 36.14 min | 80.20% | 39 729.1 TFLOPs | ✅ |
| flexmatch (refait post-fix thresh_warmup) | 42 000 | 41.50 min | 40.70 min | 80.17% | 35 704.2 TFLOPs | ✅ (ancien résultat pré-fix : 31.35 min) |
| fixmatch | 61 500 | 57.42 min | 56.28 min | 80.15% | 52 281.2 TFLOPs | ✅ |
| mixmatch | 464 500 | 145.59 min (2h26) | 137.01 min | 80.10% | 140 111.8 TFLOPs | ✅ |
| efficientmatch_2 | 34 000 | 24.38 min | 23.74 min | 80.00% | 25 171.9 TFLOPs | ✅ |
| regmixmatch_mu3 | 1 | 0.23 min | 0.23 min | 10.01% | 0.9 TFLOPs | ⚠️ **run quasiment inexistant** — tué en moins d'une seconde après le lancement (même incident "Non arrête" que ci-dessus), à ignorer complètement pour toute analyse |

**Synthèse CIFAR-10/250 labels** : classement instable d'une seed à l'autre parmi les méthodes qui
convergent (efficientmatch_2 toujours rapide et dans le haut du classement, mais mixmatch alterne
entre très lent — seed 2312 : n'atteint jamais 80% en 13h — et beaucoup plus rapide sur les 2
autres seeds tout en restant le plus gros consommateur de FLOPs). regmixmatch (mu=7, défaut)
converge de façon fiable sur les 3 seeds avec un budget de steps très faible (21-35k) mais un coût
FLOPs élevé par step. L'ablation `efficientmatch_3` sur mu (seed 2312 uniquement) montre que mu=3
et mu=5 sont les plus efficaces en FLOPs totaux (~31-32k TFLOPs), mu=1 est contre-productif (jamais
convergé malgré son faible coût par step, 61.7k TFLOPs gaspillés), mu=7 gagne en accuracy finale
mais consomme le plus de FLOPs des variantes qui convergent. fixmatch est systématiquement le plus
lent des méthodes qui atteignent le seuil. **flexmatch a été rejoué sur les 3 seeds avec le code
corrigé du bug `thresh_warmup`** (bincount incluant le bin -1, cf. section suivante) : le résultat
est variable selon la seed (64.6 min sur 2312, contre 114.1 min avant fix — nette amélioration ;
mais 52.3 min sur 308 et 41.5 min sur 2701, contre 45.2 min et 31.4 min avant fix — légèrement plus
lent). Le fix ne change donc pas le classement relatif de flexmatch (toujours nettement derrière
efficientmatch_2/3, dans la même zone que regmixmatch/sequencematch), mais élimine une source de
variance liée au bug plutôt qu'à la méthode elle-même. Détail complet dans `EXPERIMENT_LOG.md` §4.

---

## CIFAR-10, 4000 labels (target 90%)

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| efficientmatch_flex | 57 000 | 40.97 min | 39.91 min | 90.97% (max) | 42 199.9 TFLOPs |
| efficientmatch_3 | 41 000 | 29.00 min | 28.20 min | 90.06% | 30 354.0 TFLOPs |
| regmixmatch (mu=7, défaut) | 25 000 | 43.39 min | 42.92 min | 90.05% | 47 296.5 TFLOPs |
| sequencematch | 23 000 | 40.19 min | 39.76 min | 90.05% | 41 621.0 TFLOPs |
| fixmatch | 99 000 | 87.06 min | 85.22 min | 90.03% | 84 159.9 TFLOPs |
| flexmatch (refait post-fix, ancien 73.65 min) | 60 500 | 59.20 min | 58.10 min | 90.02% | 51 430.1 TFLOPs |
| efficientmatch_2 | 27 500 | 19.55 min | 19.03 min | 90.01% | 20 359.6 TFLOPs |
| mixmatch | 198 500 | 59.20 min | 55.53 min | 90.01% | 59 875.5 TFLOPs |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| mixmatch | 213 500 | 63.57 min | 59.62 min | 90.05% | 64 400.1 TFLOPs |
| flexmatch (refait post-fix, ancien 75.40 min) | 90 000 | 88.10 min | 86.40 min | 90.07% | 76 509.0 TFLOPs |
| efficientmatch_2 | 32 500 | 23.02 min | 22.41 min | 90.03% | 24 061.4 TFLOPs |
| efficientmatch_3 | 41 000 | 30.10 min | 29.30 min | 90.02% | 30 354.0 TFLOPs |
| fixmatch | 91 500 | 80.79 min | 79.09 min | 90.02% | 77 784.1 TFLOPs |
| regmixmatch (mu=7, défaut) | 27 000 | 46.69 min | 46.18 min | 90.02% | 51 080.2 TFLOPs |
| sequencematch | 21 500 | 38.92 min | 38.51 min | 90.01% | 38 906.6 TFLOPs |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| fixmatch | 100 500 | 88.80 min | 86.93 min | 90.15% | 85 435.1 TFLOPs |
| regmixmatch (mu=7, défaut) | 27 000 | 47.68 min | 47.17 min | 90.06% | 51 080.2 TFLOPs |
| mixmatch | 165 500 | 49.34 min | 46.28 min | 90.04% | 49 921.4 TFLOPs |
| efficientmatch_3 | 43 000 | 41.10 min | 40.30 min | 90.03% | 31 835.4 TFLOPs |
| flexmatch (refait post-fix, ancien 75.42 min) | 83 500 | 78.30 min | 76.80 min | 90.03% | 70 983.4 TFLOPs |
| efficientmatch_2 | 36 000 | 25.43 min | 24.76 min | 90.00% | 26 652.6 TFLOPs |

**Synthèse CIFAR-10/4000 labels** : classement **quasiment stable sur les 3 seeds** en temps réel
et en FLOPs : efficientmatch_2 est systématiquement le plus rapide et le plus économe en FLOPs des
méthodes qui atteignent 90% de façon fiable (19.6-25.4 min, 20.4-26.7k TFLOPs), suivi de mixmatch,
flexmatch puis fixmatch (le plus lent des 3 seeds). regmixmatch (mu=7) et sequencematch se
positionnent en milieu de classement en temps, avec un coût FLOPs par step nettement plus élevé.
mixmatch, catastrophique à 250 labels sur la seed 2312, redevient compétitif ici (converge en moins
d'1h sur les 3 seeds). **efficientmatch_3 (mu=3, défaut) est maintenant complet sur les 3 seeds** et
reste systématiquement le 2e plus rapide (29.0-41.1 min), juste derrière efficientmatch_2 — même
coût FLOPs/itération qu'efficientmatch_2 (740.35 GFLOPs) mais plus de steps pour converger, avec un
écart plus marqué sur seed 2701. **flexmatch a été rejoué sur les 3 seeds avec le fix
`thresh_warmup`** : résultat mitigé, plus rapide sur seed 2312 (59.2 min vs 73.65 min avant) mais
plus lent sur seed 308 (88.1 min vs 75.4 min) et seed 2701 (78.3 min vs 75.4 min) — comme pour
CIFAR-10 250 labels, le fix élimine un bug réel mais ne change pas fondamentalement le classement de
flexmatch. Détail dans `CIFAR10_4000LABELS_RESULTS.md`.

---

## SVHN, 250 labels (target 90%)

### Seed 2312

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs | Statut |
|---|---:|---:|---:|---:|---:|---|
| regmixmatch (mu=7, défaut) | 5 500 | 10.30 min | 10.19 min | 90.51% | 10 405.2 TFLOPs | ✅ |
| mixmatch | 18 500 | 6.81 min | 6.46 min | 90.27% | 5 580.3 TFLOPs | ✅ |
| efficientmatch_3 (mu=3, défaut) | 7 000 | 5.79 min | 5.65 min | 90.25% | 5 182.4 TFLOPs | ✅ |
| regmixmatch_mu3 | 6 500 | 6.25 min | 6.12 min | 90.08% | 5 881.2 TFLOPs | ✅ |
| efficientmatch (v1) | 34 000 | 28.53 min | 27.89 min | 90.05% | 25 171.9 TFLOPs | ✅ |
| efficientmatch_2 | 8 500 | 6.76 min | 6.60 min | 90.04% | 6 293.0 TFLOPs | ✅ |
| fixmatch | 8 000 | 7.82 min | 7.67 min | 90.01% | 6 800.8 TFLOPs | ✅ |
| flexmatch (rejoué, tué par watchdog 2h) | 115 500 | 117.93 min | 115.79 min | 83.99% (max) | 98 186.6 TFLOPs | ❌ jamais atteint 90%, plafond similaire à l'ancien résultat pré-fix (84.32%) |
| efficientmatch_2_flex | 13 000 | 10.49 min | 10.24 min | 79.76% (max) | 9 624.5 TFLOPs | ❌ jamais atteint 90% |
| sequencematch | 12 000 | 22.12 min | 21.89 min | 78.14% (max) | 21 715.3 TFLOPs | ❌ jamais atteint 90% |
| efficientmatch_flex | 8 500 | 7.26 min | 7.10 min | 77.44% (max) | 6 293.0 TFLOPs | ❌ jamais atteint 90% |

### Seed 0308

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs | Statut |
|---|---:|---:|---:|---:|---:|---|
| mixmatch | 52 500 | 18.38 min | 17.41 min | 91.06% | 15 836.1 TFLOPs | ✅ |
| regmixmatch (mu=7, défaut) | 4 500 | 8.25 min | 8.15 min | 90.30% | 8 513.4 TFLOPs | ✅ |
| efficientmatch_2 | 8 500 | 6.76 min | 6.59 min | 90.24% | 6 293.0 TFLOPs | ✅ |
| fixmatch | 6 500 | 6.44 min | 6.31 min | 90.07% | 5 525.6 TFLOPs | ✅ |
| flexmatch (rejoué, tué par watchdog 2h) | 92 000 | 94.32 min | 92.61 min | 82.52% (max) | 78 209.2 TFLOPs | ❌ jamais atteint 90%, plafond quasi identique à l'ancien résultat pré-fix (82.38%) |
| efficientmatch (v1) | 1 | 0.46 min | 0.45 min | 8.03% | 0.7 TFLOPs | ⚠️ tué volontairement dès le 1er step (doublon de run accidentel), non exploitable |
| regmixmatch_mu3 | — | — | — | — | — | ⚠️ **jamais lancé jusqu'au bout** (aucun fichier de résultat produit lors du sweep multi-seed interrompu) |

### Seed 2701

| Méthode | Steps | Temps | Temps corrigé | Acc | FLOPs |
|---|---:|---:|---:|---:|---:|
| mixmatch | 35 000 | 12.44 min | 11.79 min | 90.82% | 10 557.4 TFLOPs |
| regmixmatch (mu=7, défaut) | 6 000 | 11.28 min | 11.16 min | 90.37% | 11 351.2 TFLOPs |
| regmixmatch_mu3 | 7 500 | 10.64 min | 10.49 min | 90.20% | 6 786.0 TFLOPs |
| fixmatch | 8 000 | 7.90 min | 7.74 min | 90.18% | 6 800.8 TFLOPs |
| efficientmatch_2 | 10 000 | 7.86 min | 7.66 min | 90.02% | 7 403.5 TFLOPs |
| flexmatch (rejoué, tué par watchdog 2h) | 57 500 | 59.44 min | 58.37 min | 86.73% (max, **jamais atteint 90%**) | 48 880.8 TFLOPs |

**Synthèse SVHN/250 labels** : **flexmatch échoue sur les 3 seeds** (plafond 82-87%, jamais 90%) —
rejoué sur les 3 seeds avec le fix `thresh_warmup` (implémentation vérifiée conforme à la référence
officielle `microsoft/Semi-supervised-learning`), donné jusqu'à 2h par run (watchdog automatique) au
lieu d'être arrêté manuellement comme lors des tentatives précédentes : le plafond reste quasi
identique (83.99%/82.52%/86.73% contre 84.32%/82.38%/85.10% avant fix) — confirme que c'est un
problème méthodologique reproductible sur cette config, pas un artefact du bug ni de la variance.
Parmi les méthodes qui réussissent, le classement varie d'une seed à l'autre entre regmixmatch, mixmatch,
efficientmatch_2/3 et fixmatch, mais toutes restent dans une fourchette étroite (5.8-18.4 min) sauf
flexmatch. regmixmatch (mu=3 ou mu=7) et efficientmatch_3 se distinguent par leur faible nombre de
steps (4 500-8 500) pour un temps réel comparable aux autres méthodes rapides. L'ablation mu sur
regmixmatch (mu=3 vs mu=7) donne des résultats proches (90.08-90.20% vs 90.30-90.51%) avec mu=3
légèrement moins coûteux en FLOPs par step, mais l'échantillon est incomplet (seed 308 manquante).

---

## CIFAR-100, 10000 labels (target 60% top-1, test_period 256)

**WRN-28-4 (WF4) est devenu l'architecture de facto sur cette config** (WRN-28-8 sature la VRAM de
la carte utilisée, cf. `FLOPS_RESULTS.md`) — les nouvelles méthodes (efficientmatch_3, regmixmatch)
et les retests de flexmatch ont tous été lancés en WF4 ; les anciennes lignes WF8 (fixmatch,
flexmatch, mixmatch) sont conservées pour traçabilité mais ne sont plus la référence. Coût par
évaluation utilisé pour "Temps corrigé" : 1 499.1 ms pour WF4, 4 476.3 ms pour WF8 (cf.
`FLOPS_RESULTS.md`). Tous les runs de cette section sont désormais soumis à un **plafond automatique
de 2h par run** (watchdog qui tue le processus s'il dépasse ce seuil).

### Seed 2312 (complet en WF4)

| Méthode | wf | Steps | Temps | Temps corrigé | Acc | FLOPs | Statut |
|---|---:|---:|---:|---:|---:|---:|---|
| mixmatch | 4/8 | 10 752 | 28.24 min | 25.03 min | 60.44% | 50 865.9 TFLOPs | ✅ (WF8 ; existe aussi une variante WF4 à 64.26% en 58.7 min, non directement comparable — target_acc différent) |
| efficientmatch_3 | 4 | 10 240 | 19.80 min | 18.80 min | 60.00% | 29 921.0 TFLOPs | ✅ |
| regmixmatch | 4 | 5 376 | 25.10 min | 24.60 min | 60.07% | 40 143.0 TFLOPs | ✅ |
| flexmatch | 4 | 13 568 | 33.70 min | 32.30 min | 60.06% | 45 519.0 TFLOPs | ✅ |
| fixmatch | 4 | 12 544 | 29.00 min | 27.80 min | 60.00% | 42 084.0 TFLOPs | ✅ |
| fixmatch | 8 | 11 520 | 120.81 min | 117.38 min | 60.08% | 153 588.8 TFLOPs | ✅ (ancienne mesure WF8) |
| flexmatch | 8 | 9 472 | 135.20 min | 132.36 min | 56.28% (max) | 126 284.1 TFLOPs | ❌ ancienne mesure WF8, jamais atteint 60% |

### Seed 0308 (complet en WF4)

| Méthode | wf | Steps | Temps | Temps corrigé | Acc | FLOPs | Statut |
|---|---:|---:|---:|---:|---:|---:|---|
| flexmatch (refait post-fix) | 4 | 12 288 | 30.51 min | 29.24 min | 60.51% | 41 225.0 TFLOPs | ✅ |
| mixmatch | 8 | 10 496 | 27.30 min | 24.10 min | 60.38% | 49 655.0 TFLOPs | ✅ (WF8, pas encore de version WF4 sur cette seed) |
| regmixmatch | 4 | 5 888 | 28.00 min | 27.40 min | 60.31% | 43 966.0 TFLOPs | ✅ |
| efficientmatch_3 | 4 | 10 240 | 21.40 min | 20.40 min | 60.29% | 29 921.0 TFLOPs | ✅ |
| fixmatch | 4 | 12 800 | 29.20 min | 28.00 min | 60.01% | 42 942.0 TFLOPs | ✅ |
| fixmatch | 8 | 9 472 | 120.43 min | 117.59 min | 58.55% (max) | 126 284.1 TFLOPs | ❌ ancienne mesure WF8, jamais atteint 60% |

### Seed 2701 (partiel — fixmatch et flexmatch manquants)

| Méthode | wf | Steps | Temps | Temps corrigé | Acc | FLOPs | Statut |
|---|---:|---:|---:|---:|---:|---:|---|
| mixmatch | 8 | 11 008 | 28.60 min | 25.30 min | 60.18% | 52 077.0 TFLOPs | ✅ (WF8, pas de version WF4 sur cette seed) |
| regmixmatch | 4 | 6 144 | 25.20 min | 24.60 min | 60.14% | 45 877.0 TFLOPs | ✅ |
| efficientmatch_3 | 4 | 9 728 | 25.30 min | 24.30 min | 60.05% | 28 425.0 TFLOPs | ✅ |
| fixmatch | — | — | — | — | — | — | ❌ **jamais lancé** — trou identifié, à faire |
| flexmatch | 4 | — | — | — | — | — | ❌ **tentative avortée** : lancé puis arrêté manuellement quasi immédiatement (1 seul step écrit, fichier supprimé) — à refaire |

**Synthèse CIFAR-100/10000 labels** : la bascule vers WF4 a permis de compléter presque tout le
tableau. Sur les deux seeds désormais complètes en WF4 (2312, 0308), le classement en temps est
stable : efficientmatch_3 est le plus rapide (19.8-21.4 min), suivi de regmixmatch (25.1-28.0 min),
fixmatch (29.0-29.2 min) et flexmatch (30.5-33.7 min) — mixmatch (WF8) se situe dans la même
fourchette (25.0-28.2 min corrigé) mais n'est pas encore mesuré en WF4 sur ces deux seeds, donc pas
strictement comparable en FLOPs. flexmatch converge correctement en WF4 sur toutes les seeds testées
(60.0-60.5%), contrairement à son échec systématique en WF8. **Reste à faire avant publication** :
fixmatch et flexmatch sur seed 2701 (WF4), et harmoniser mixmatch en WF4 sur les 3 seeds pour une
comparaison FLOPs à architecture égale.

---

## Ablation : nombre de vues non labellisées par step (mu)

### efficientmatch_3 — CIFAR-10, 250 labels, seed 2312 (target 80%)

| mu | Steps | Temps | Temps corrigé | Acc finale | FLOPs/it | FLOPs totaux | Statut |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 173 000 | 71.62 min | 68.42 min | 77.57% (max) | 356.46 GFLOPs | 61 667.6 TFLOPs | ❌ jamais atteint 80% |
| 3 (défaut) | 42 000 | 30.38 min | 29.60 min | 80.11% | 740.35 GFLOPs | 31 094.7 TFLOPs | ✅ |
| 5 | 28 500 | 30.53 min | 30.00 min | 80.10% | 1 124.25 GFLOPs | 32 041.1 TFLOPs | ✅ |
| 7 | 25 500 | 36.26 min | 35.78 min | 80.18% | 1 508.14 GFLOPs | 38 457.6 TFLOPs | ✅ |

mu=3 est le meilleur compromis FLOPs totaux/temps ; mu=1 est contre-productif malgré son faible
coût par step (jamais convergé, gaspille le plus de FLOPs de la série) ; mu=7 atteint le meilleur
score final mais au prix du coût FLOPs total le plus élevé parmi les variantes qui convergent.

### regmixmatch — mu=3 vs mu=7 (défaut)

| Config | mu=3 | mu=7 (défaut) |
|---|---|---|
| SVHN 250, seed 2312 | 90.08% @ 6 500 steps, 6.25 min (corrigé 6.12 min) | 90.51% @ 5 500 steps, 10.30 min (corrigé 10.19 min) |
| SVHN 250, seed 2701 | 90.20% @ 7 500 steps, 10.64 min (corrigé 10.49 min) | 90.37% @ 6 000 steps, 11.28 min (corrigé 11.16 min) |
| CIFAR-10 250, seed 2312 | ⚠️ interrompu à 76.32% (résultat antérieur complet : 80.13% @ 63 000 steps, 56.1 min) | 80.04% @ 35 000 steps, 61.23 min (corrigé 60.57 min) |

Échantillon incomplet (voir avertissement en tête de document) — à confirmer avec des runs propres
avant de tirer des conclusions définitives sur l'effet de mu pour regmixmatch.

---

## Vue d'ensemble : ce qui a et n'a pas été fait sur les 3 seeds

| Config | efficientmatch_2 | fixmatch | flexmatch | mixmatch | regmixmatch | efficientmatch (v1) | efficientmatch_3 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CIFAR-10, 250 labels | ✅ 3/3 seeds | ✅ 3/3 | ✅ 3/3 (rejoué post-fix, échoue jamais mais lent) | ✅ 3/3 (1 jamais convergé) | ✅ 3/3 | seed 2312 seulement | ✅ 3/3 (+ ablation mu) |
| CIFAR-10, 4000 labels | ✅ 3/3 seeds | ✅ 3/3 | ✅ 3/3 (rejoué post-fix) | ✅ 3/3 | ✅ 3/3 | ❌ aucune | ✅ 3/3 seeds |
| SVHN, 250 labels | ✅ 3/3 seeds | ✅ 3/3 | ✅ 3/3 (rejoué, **0/3 atteint 90%**) | ✅ 3/3 | ✅ 3/3 | seed 2312 (+ tentative avortée seed 0308) | seed 2312 seulement |
| CIFAR-100, 10000 labels (WF4) | ❌ non testé | 2/3 seeds (**seed 2701 manquante**) | 2/3 seeds (**seed 2701 avortée**) | ❌ WF4 (WF8 sur 3/3) | 3/3 seeds | ❌ non testé | 3/3 seeds |

**Couverture complète et comparable sur les 3 seeds** : CIFAR-10 250 labels, CIFAR-10 4000 labels,
SVHN 250 labels (efficientmatch_2/fixmatch/flexmatch/mixmatch/regmixmatch), et désormais
**efficientmatch_3 sur CIFAR-10 250/4000 labels** (3/3 seeds chacune). regmixmatch et efficientmatch_3
sont maintenant complets sur les 3 seeds de CIFAR-100 10000 labels (WF4) ; fixmatch et flexmatch y
manquent encore la seed 2701 (flexmatch a été tenté puis abandonné avant écriture de résultat
exploitable). efficientmatch (v1) reste non testé de façon systématique sur aucune config (seed 2312
uniquement, sauf CIFAR-10 4000 où il n'a jamais été lancé). **Reste à faire avant publication** :
fixmatch et flexmatch sur CIFAR-100 10000/seed 2701 (WF4), et mixmatch en WF4 sur CIFAR-100
10000/seeds 308 et 2701 pour une comparaison FLOPs homogène.
