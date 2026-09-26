# Expérience asymptotique : efficientmatch vs regmixmatch sans limite d'accuracy

Objectif : voir jusqu'où monte chaque méthode quand on ne l'arrête pas à une accuracy cible (contrairement à
`EXPERIENCE_PRINCIPALE.md`, où chaque run s'arrête à `target_acc`), et comparer les deux méthodes **à accuracy
égale** (temps et FLOPs pour atteindre chaque palier) et **à temps égal**.

## Protocole

- **Configuration** : CIFAR-10, 250 labels, WRN-28-2, **seed 42 uniquement** (une seule seed).
- **Méthodes** : `efficientmatch_3` (mu=3, EMA) et `regmixmatch` (mu=7 par défaut, EMA).
- **Sans `--target_acc`** : horizon par défaut de 2^20 = 1 048 576 steps, pas de watchdog de 2h, évaluation tous
  les 500 steps. Les résultats sont écrits sous des noms particuliers (option `--tag unlimited`) :
  `results/cifar10-labeled-250-seed-42/efficientmatch_3_ema_unlimited_metrics.json` et
  `results/cifar10-labeled-250-seed-42/regmixmatch_ema_unlimited_metrics.json`.
- **Temps corrigé** = temps brut − (nombre d'évaluations × 0.5537 s), coût d'une évaluation en WRN-28-2
  (cf. `FLOPS_RESULTS.md`). **FLOPs** = steps × FLOPs/itération : 740.35 GFLOPs (efficientmatch_3, mu=3) et
  1 891.86 GFLOPs (regmixmatch, mu=7).
- Les "paliers" sont les **premiers franchissements** d'une accuracy par l'accuracy de test évaluée (EMA).

## Statut des deux runs

| | efficientmatch_3 | regmixmatch |
|---|---|---|
| Steps réalisés | 1 048 576 (horizon complet) | **418 500 (arrêtée à la main, 40% de l'horizon)** |
| Temps de run | 749.4 min (12.5 h) | 750.6 min (12.5 h) |
| Évaluations | 2 099 | 838 |
| Accuracy à la dernière évaluation | 91.25% | 90.33% |
| **Meilleure accuracy** | **91.62%** (step 1 026 000, 733 min) | **90.69%** (step 418 000, 750 min) |
| Vitesse | ≈ 43 ms/step | ≈ 108 ms/step |

**Regmixmatch n'a pas été menée à son terme** : elle a été arrêtée après 12.5 h (environ 30 h auraient été
nécessaires pour les 1 048 576 steps). Sa courbe de learning rate (cosinus sur 2^20 steps) n'était donc qu'à
40% de son annealing : son accuracy asymptotique n'est **pas** mesurée, et la comparaison à temps égal ci-dessous
compare deux runs qui n'en sont pas à la même phase de la décroissance du learning rate (efficientmatch a
terminé son annealing, regmixmatch non).

Deux tentatives précédentes ont été interrompues et écartées : une run efficientmatch seed 2312 (arrêtée à
13 000 steps, 69.72%) et une première run regmixmatch seed 42 (morte silencieusement au step 4 000, relancée
depuis zéro).

## Comparaison à accuracy égale (premier franchissement)

| Palier | efficientmatch : steps | corrigé | TFLOPs | regmixmatch : steps | corrigé | TFLOPs | Avance en temps | Facteur temps | Facteur FLOPs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70% | 9 500 | 7.3 min | 7 033 | 9 000 | 16.5 min | 17 027 | 9 min | ×2.26 | ×2.42 |
| 75% | 14 000 | 10.5 min | 10 365 | 15 000 | 27.3 min | 28 378 | 17 min | ×2.60 | ×2.74 |
| 80% | 29 000 | 20.8 min | 21 470 | 28 000 | 50.1 min | 52 972 | 29 min | ×2.41 | ×2.47 |
| 82% | 46 000 | 32.4 min | 34 056 | 41 500 | 73.7 min | 78 512 | 41 min | ×2.27 | ×2.31 |
| 84% | 74 000 | 51.7 min | 54 786 | 62 000 | 109.6 min | 117 295 | 58 min | ×2.12 | ×2.14 |
| 85% | 102 000 | 71.3 min | 75 516 | 77 000 | 136.0 min | 145 673 | 65 min | ×1.91 | ×1.93 |
| 86% | 160 000 | 111.1 min | 118 456 | 117 000 | 206.6 min | 221 348 | 96 min | ×1.86 | ×1.87 |
| 87% | 210 500 | 146.2 min | 155 844 | 174 500 | 308.3 min | 330 130 | 162 min | ×2.11 | ×2.12 |
| 88% | 291 500 | 201.7 min | 215 812 | 239 000 | 422.5 min | 452 155 | 221 min | ×2.09 | ×2.10 |
| 89% | 429 500 | 297.7 min | 317 980 | 250 500 | 443.0 min | 473 911 | 145 min | ×1.49 | ×1.49 |
| 90% | 684 500 | 475.4 min | 506 770 | 306 500 | 542.3 min | 579 855 | 67 min | ×1.14 | ×1.14 |
| 90.3% | 741 000 | 515.2 min | 548 599 | 358 500 | 634.7 min | 678 232 | 120 min | ×1.23 | ×1.24 |
| 90.5% | 789 500 | 549.4 min | 584 506 | 392 500 | 695.6 min | 742 555 | 146 min | ×1.27 | ×1.27 |
| 91% | 884 000 | 616.1 min | 654 469 | jamais atteint | | | | | |
| 91.5% | 1 000 000 | 696.4 min | 740 350 | jamais atteint | | | | | |

Les facteurs temps et FLOPs sont quasi identiques, car le temps par step est proportionnel au coût en FLOPs
pour ces deux méthodes (≈ 2.4x plus cher par step pour regmixmatch).

**Lecture** :
- Sur les paliers de 70% à 88%, efficientmatch atteint l'accuracy visée environ **2 fois plus vite** (×1.9 à
  ×2.6), soit de 9 min d'avance à 70% jusqu'à 221 min à 88%, et consomme environ 2 fois moins de FLOPs.
- L'avantage se réduit à haute accuracy (×1.5 à 89%, ×1.1 à 90%, ×1.2-1.3 entre 90.3% et 90.5%) : regmixmatch
  rattrape en nombre de steps (il atteint 90% en 306 500 steps contre 684 500), mais pas en temps.
- **Prudence à partir de 89%** : les courbes oscillent d'environ ±0.3 point, donc l'instant du premier
  franchissement d'un palier peut varier de plusieurs dizaines de minutes d'une évaluation à l'autre ; les
  avances de 145, 67, 120 et 146 min à 89-90.5% sont peu fiables.

## Comparaison à temps égal (temps brut)

| Temps | efficientmatch (step, acc, meilleur jusque-là) | regmixmatch (step, acc, meilleur jusque-là) |
|---:|---|---|
| 10 min | 12 500, 73.67%, 73.67% | 5 500, 62.25%, 62.25% |
| 30 min | 41 000, 81.62%, 81.70% | 16 000, 75.74%, 75.74% |
| 60 min | 83 500, 84.24%, 84.55% | 33 000, 81.31%, 81.31% |
| 120 min | 168 000, 86.11%, 86.31% | 67 000, 84.08%, 84.60% |
| 240 min | 338 000, 87.90%, 88.28% | 134 500, 85.97%, 86.27% |
| 360 min | 504 000, 88.85%, 89.14% | 201 500, 87.02%, 87.28% |
| 480 min | 673 000, 89.82%, 89.94% | 268 500, 88.90%, 89.20% |
| 600 min | 839 000, 90.40%, 90.70% | 335 000, 89.56%, 90.26% |
| 720 min | 1 007 000, 91.45%, 91.57% | 401 500, 90.15%, 90.56% |
| 750 min | 1 048 576, 91.25%, 91.62% | 418 000, 90.69%, 90.69% |

L'écart d'accuracy à temps égal se réduit de ≈ 11 points à 10 min à ≈ 3 points à 60 min, puis ≈ 1 point de
480 min à 750 min (91.62% contre 90.69% en meilleur point).

## Comparaison à nombre de steps égal

| Steps | efficientmatch | regmixmatch |
|---:|---:|---:|
| 5 000 | 53.51% | 60.13% |
| 20 000 | 77.72% | 77.69% |
| 50 000 | 82.48% | 82.56% |
| 100 000 | 84.82% | 85.54% |
| 200 000 | 86.68% | 87.23% |
| 300 000 | 87.73% | 89.20% |
| 400 000 | 88.28% | 90.08% |
| 418 500 | 88.44% | 90.33% |

À nombre de steps égal, **regmixmatch est légèrement meilleur** à partir de 100 000 steps (+1.9 point à
418 500 steps). Son désavantage en temps vient uniquement de son coût par step (≈ 2.4x). Attention : à ce
nombre de steps, efficientmatch est encore en phase de learning rate élevé, alors que son horizon est
plus loin ; l'inverse vaut pour la fin de run de chaque méthode.

## Conclusions

1. **efficientmatch_3 plafonne autour de 91-92% sur CIFAR-10 250 labels** (meilleur point 91.62%, moyenne des
   10 dernières évaluations 91.25%), avec une montée très lente après 90% (60 min de temps corrigé
   supplémentaires par demi-point). Les 80% habituels de l'expérience principale sont atteints en 21 min.
2. **À accuracy égale, efficientmatch est environ 2 fois plus rapide et plus économe en FLOPs que regmixmatch**
   sur les paliers de 70% à 88% ; l'avantage se resserre à haute accuracy.
3. **À steps égaux, regmixmatch est légèrement meilleur** : l'efficacité d'efficientmatch tient à son coût par
   itération plus faible, pas à un meilleur gain par step.
4. **Limites** : une seule seed, regmixmatch arrêtée à 40% de l'horizon (asymptote non mesurée), bruit des
   premiers franchissements à haute accuracy.

## Figures

Générées par `scripts/plot_acc_vs_time.py` avec `--no_budget --hide_target` (courbes non tronquées à 2h, sans
ligne de cible), y ∈ [0.70, 0.93] :

- `figures/unlimited_efficientmatch_vs_regmixmatch_acc_vs_steps_cifar10_250_seed42.png`
- `figures/unlimited_efficientmatch_vs_regmixmatch_acc_vs_time_cifar10_250_seed42.png`
- `figures/unlimited_efficientmatch_vs_regmixmatch_acc_vs_flops_cifar10_250_seed42.png`

## Reproduire

```bash
cd scripts
python efficientmatch_3.py --dataset cifar10 --num_labeled 250 --seed 42 --tag unlimited
python regmixmatch.py --dataset cifar10 --num_labeled 250 --seed 42 --tag unlimited
```
