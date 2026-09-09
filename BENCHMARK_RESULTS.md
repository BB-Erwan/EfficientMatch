# Résultats du benchmark de FLOPs et de vitesse

Mesures réalisées via `scripts/benchmark.py` le 2026-09-09, sur NVIDIA RTX A4000 (`torch==2.8.0+cu129`, Python 3.11.13).

**Baseline volontairement pessimiste** : toutes les optimisations de vitesse sont désactivées ici
(`--no-compile --no-amp --no-pin-memory --num-workers 0`) -- ces temps sont donc un **majorant**, pas
le temps réel d'un run avec les réglages par défaut de chaque script (`--compile`/`--amp`/
`--pin-memory` activés par défaut). Cohérent avec la philosophie du protocole (PROJECT_SPEC.md §1) :
les FLOPs sont la métrique d'efficacité indépendante du hardware ; le temps d'horloge dépend fortement
de la pile logicielle (compilateur, précision, etc.), d'où l'intérêt d'avoir aussi un point de mesure
sans aucune de ces optimisations.

## Configuration de la mesure

| Paramètre | Valeur |
|---|---|
| Dataset | CIFAR-10 |
| n_labels | 250 |
| Sous-ensemble non labellisé (mesure uniquement) | 4096 |
| Budget cible K (pour l'extrapolation) | 131072 |
| Itérations de chauffe / mesurées | 20 / 100 |
| `torch.compile` | désactivé |
| `torch.autocast` (bfloat16) | désactivé |
| `pin_memory` / `non_blocking` | désactivé |
| `num_workers` (DataLoader) | 0 |

Les FLOPs sont mesurés une seule fois par algo (`torch.utils.flop_counter.FlopCounterMode`, forward + backward) sur un modèle jetable et des tenseurs factices de la bonne forme -- indépendants des vraies données, donc indépendants aussi de `--compile`/`--amp`.
Le temps par itération est chronométré INDIVIDUELLEMENT sur chacune des itérations mesurées (boucle réelle, mêmes appels que `main()`, CIFAR-10 réel, sans EMA ni logging/évaluation) plutôt qu'en moyenne globale sur tout le bloc -- l'écart-type reporté permet de voir si la mesure est stable (le tout début de l'entraînement peut être irrégulier : recompilation tardive d'une branche pas vue pendant la chauffe, effets de cache). Toutes les mesures ci-dessous ont un coefficient de variation ≤ 11 %, donc stables sur cette fenêtre.

## Résultats

| Méthode | FLOPs / itération | ms / itération | it/s | Temps total estimé (K=131072) |
|---|---:|---:|---:|---:|
| efficientmatch | 7.404e+11 | 564.6 ± 26.8 | 1.77 | 20.56 h |
| fixmatch | 8.501e+11 | 1099.9 ± 53.1 | 0.91 | 40.05 h |
| flexmatch | 8.501e+11 | 1109.7 ± 44.2 | 0.90 | 40.40 h |
| mixmatch | 3.016e+11 | 214.2 ± 22.6 | 4.67 | 7.80 h |

Total séquentiel pour les 4 algorithmes ci-dessus (1 graine, K=131072) : **108.8 h**.

Remarques :
- **EfficientMatch a moins de FLOPs et va plus vite que FixMatch/FlexMatch**, alors qu'il fait un
  forward supplémentaire (canal Mixup) par itération -- ceci vient de `mu=3` par défaut pour
  EfficientMatch contre `mu=7` pour FixMatch/FlexMatch (moins d'exemples non labellisés par itération
  compense largement le canal Mixup additionnel).
- **MixMatch est nettement le plus rapide et le moins coûteux en FLOPs** des quatre : `mu=1` par
  défaut (contre 3 ou 7 pour les autres) et pas de vue forte (RandAugment), donc un batch total par
  itération beaucoup plus petit.
- Ces quatre méthodes n'utilisent pas des `mu` identiques par défaut : une comparaison à protocole
  strictement égal (même `mu` partout) donnerait des temps/FLOPs différents -- cf. `--mu` de chaque
  script pour rejouer la mesure à budget de calcul comparable.

## Reproduire

```powershell
python scripts/benchmark.py --n-labels 250 --K 131072 --no-compile --no-amp --no-pin-memory --num-workers 0 --out BENCHMARK_RESULTS.md
```

Pour mesurer avec les optimisations de vitesse par défaut (`--compile --amp --pin-memory`, qui sont
déjà les valeurs par défaut de chaque script) :

```powershell
python scripts/benchmark.py --n-labels 250 --K 131072 --out BENCHMARK_RESULTS_optimized.md
```
