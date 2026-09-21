# Fast FixMatch (CBS) : gain FLOPs vs. surcoût de temps d'horloge

Diagnostic ciblé (pas une comparaison de convergence à 3 seeds) destiné à quantifier l'affirmation
du papier selon laquelle le Curriculum Batch Size (CBS) de Fast FixMatch \citep{chen2024fast},
malgré un gain FLOPs réel, se traduit dans notre pipeline par un surcoût de temps d'horloge qui le
neutralise en pratique -- à cause des formes de tenseur non fixes d'une itération à l'autre.

## Protocole

- **Nouveau script `scripts/fast_fixmatch.py`** : copie fidèle de `fixmatch.py`, avec exactement
  deux différences :
  1. La taille du batch non étiqueté n'est **pas fixe** (`mu * batch_size_l`) mais suit le
     schedule **B-EXP** du papier de référence (arXiv:2309.03469) :
     `B(t) = u * (1 - (1 - t/T) / ((1 - alpha) + alpha * (1 - t/T)))`, `alpha = 0.7`, plancher
     `cbs_min_batch = 8`. Le batch complet (`mu * batch_size_l`) continue d'être tiré du
     DataLoader existant à chaque step, puis simplement tronqué à la taille prescrite par la
     courbe pour ce step -- pas de DataLoader à taille variable.
  2. Le poids de la loss non supervisée est `lambda = batch_u_actuel / batch_size_l` (règle du
     papier), au lieu du poids implicite de 1.0 dans `fixmatch.py`.
  - L'horizon de la courbe CBS (`--cbs_total_steps`) est **volontairement découplé** de
    `--total_steps` (l'horizon nominal surdimensionné du LR schedule, 2^20, partagé par tous les
    scripts de ce repo) -- coupler les deux aurait reproduit le bug déjà rencontré et corrigé sur
    `mixmatch_2.py` (rampe de `lambda_u` qui ne progresse quasiment jamais sur la durée réelle d'un
    run ici).
- **Pipeline standard, pas de version "sans optimisation"** : `torch.compile` (mode
  `reduce-overhead`), `channels_last`, autocast bfloat16, `cudnn.benchmark=True` -- exactement les
  réglages utilisés pour produire les résultats principaux (Tab.~\ref{tab:main_results}), puisque
  c'est précisément "notre pipeline" que le paragraphe met en cause.
- **Une seule run courte**, pas de course à la convergence : CIFAR-10, 250 labels, seed 2312,
  **3000 steps**, `--cbs_total_steps 3000` -- la courbe CBS balaie une fois l'intégralité de sa
  plage (8 -> 448) sur ces 3000 steps, ce qui suffit à observer l'effet des formes de tenseur
  variables sans avoir besoin d'un entraînement complet.
- **Pas de nouvelle run FixMatch** : comparaison contre le `fixmatch_ema_metrics.json` déjà en
  base sur cette même config/seed (résultat de l'expérience principale), tronqué à son point de
  contrôle le plus proche de 3000 steps.
- **FLOPs de Fast FixMatch** : calibrés une fois par step, via deux mesures `FlopCounterMode`
  (forward seul et forward+backward, à `batch_size_l=64` de référence) -- les FLOPs d'un
  forward/backward CNN classique scalent exactement linéairement avec la taille de batch, donc ces
  deux mesures suffisent à calculer le coût exact de n'importe quelle taille de batch rencontrée
  pendant la courbe, sans avoir à remesurer à chaque forme. Cumulés step par step dans les
  métriques (`cumulative_gflops`).
- **FLOPs de FixMatch** : `850.10` GFLOPs/itération (constant, `WRN-28-2`, mesuré séparément dans
  `FLOPS_RESULTS.md` / `scripts/flops_analysis.py`), multiplié par le nombre de steps.

## Résultat

À step=3000 (identique pour les deux méthodes) :

| Méthode | Temps d'horloge | GFLOPs cumulés | ms/itération |
|---|---:|---:|---:|
| FixMatch (batch fixe = 448) | 197.9 s | 2 550 300 | 65.96 |
| Fast FixMatch (CBS, batch 8 -> 448) | 421.0 s | 953 154 | 140.32 |
| **Ratio (Fast FixMatch / FixMatch)** | **x2.13** | **x0.37** | **x2.13** |

- **Gain FLOPs confirmé** : Fast FixMatch utilise 62.6% de FLOPs en moins que FixMatch sur les
  mêmes 3000 steps -- cohérent avec le gain rapporté par les auteurs.
- **Surcoût de temps d'horloge confirmé, et il neutralise largement le gain FLOPs** : dans notre
  pipeline, Fast FixMatch est **2.13x plus lent en temps réel** que FixMatch sur les mêmes 3000
  steps, alors même qu'il traite moins d'échantillons. Le log d'exécution attribue explicitement ce
  surcoût aux formes de tenseur variables : `torch._inductor` (CUDA Graphs, mode
  `reduce-overhead`) émet un avertissement dès la 2e forme d'entrée rencontrée
  (`"CUDAGraph supports dynamic shapes by recording a new graph for each distinct input size...
  We have observed 9 distinct sizes"`), confirmant que chaque nouvelle taille de batch déclenche un
  nouvel enregistrement de graphe CUDA plutôt que la réutilisation d'un graphe déjà compilé -- le
  mécanisme exact décrit dans le paragraphe (implémentation compliquée par les formes non fixes).

## Conclusion pour le papier

Ces chiffres remplacent le `\textcolor{orange}{à confirmer quantitativement}` : dans notre
pipeline, le CBS de Fast FixMatch entraîne un **surcoût de temps d'horloge d'un facteur ~2.1x**,
qui neutralise (et inverse) le gain FLOPs de ~2.7x observé sur le papier de trafic (0.37x de
FLOPs). Le gain de \citep{chen2024fast} reste valide sur la métrique FLOPs -- celle que nous
retenons comme critère principal indépendant du hardware -- mais ne se traduit pas par un gain de
temps réel dans notre pile logicielle (CUDA Graphs incompatibles avec des formes de tenseur
variables sans repadding). Il s'agit d'une limite de notre implémentation/pipeline, pas d'une
remise en cause du résultat original des auteurs.
