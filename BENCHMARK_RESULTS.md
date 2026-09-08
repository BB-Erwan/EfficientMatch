# Phase 1 — Résultats du benchmark de vitesse

Mesures réalisées sur la machine cible (NVIDIA RTX A4000, `torch==2.8.0+cu129`,
`triton-windows==3.4.0.post21`) via `scripts/benchmark_speed.py`, le 2026-09-08.

## Configuration de la mesure

| Paramètre | Valeur |
|---|---|
| Dataset | CIFAR-10 |
| n_labels | 250 |
| Sous-ensemble non labellisé (mesure uniquement) | 4096 |
| Budget cible K (pour l'extrapolation) | 2¹⁷ = 131072 |
| Itérations de chauffe / mesurées | 10 / 30 |
| `torch.compile` | activé (mode par défaut, PAS `reduce-overhead`) pour FixMatch/FlexMatch/MixMatch/EfficientMatch ; désactivé pour Fast FixMatch (garde-fou CBS, cf. plus bas) |

## Résultats

| Méthode | it/s (ou débit effectif) | Temps total estimé (K=2¹⁷) |
|---|---:|---:|
| FixMatch | 7.47 | **4.88 h** |
| FlexMatch | 7.63 | **4.77 h** |
| MixMatch | 6.05 | **6.02 h** |
| EfficientMatch | 5.98 | **6.09 h** |
| Fast FixMatch | 3.85 (effectif) | **9.47 h** |

Total séquentiel pour un passage à 1 graine sur les 4 méthodes prioritaires (RQ1+RQ2, hors Fast
FixMatch) : **~21.8 h**. Avec Fast FixMatch inclus (annexe) : **~31.2 h**. Compter ×3 pour 3 graines
si le temps le permet.

## Fast FixMatch : plus lent en temps réel malgré moins de FLOPs

Point notable, découvert en creusant une anomalie de mesure (0.32 it/s / 112h dans une première
version buguée du benchmark) : **Fast FixMatch est ~2× plus lent en temps d'horloge que FixMatch
compilé, sur cette machine, alors que c'est l'algorithme le plus économe en FLOPs des cinq.**
Économie de calcul (FLOPs) et vitesse réelle (temps d'horloge) sont deux choses différentes ici, pour
trois raisons propres à l'implémentation du Curriculum Batch Size (CBS) :

1. **Pas de `torch.compile` du tout sur l'ensemble du run.** La taille du batch non labellisé change
   à chaque itération (croît de `cbs_min_batch=8` à `mu*B=448` au cours du run) ; compiler dans ces
   conditions déclencherait une recompilation quasi permanente au lieu d'une accélération (bug
   observé et documenté sur cette machine). Fast FixMatch n'a donc jamais accès au ×2.9 de speedup
   que `torch.compile` apporte aux 4 autres méthodes.
2. **Un batch petit est moins efficace par échantillon sur GPU.** Le coût fixe d'une itération
   (lancement des kernels, dispatch Python, étape d'optimiseur, mise à jour EMA...) reste à peu près
   constant que le batch non labellisé fasse 8 ou 448 exemples. En début de run (batch petit), ce
   coût fixe est amorti sur très peu d'exemples : moins efficace par échantillon qu'en fin de run.
3. **Recalcul cuDNN à chaque nouvelle taille de batch rencontrée** (avec `cudnn_benchmark=True`) : au
   plus ~441 tailles distinctes entre `cbs_min_batch` et `mu*B` sur un run complet, donc jusqu'à 441
   re-recherches d'algorithme ponctuelles -- un coût que les 4 autres algos (taille de batch fixe) ne
   paient qu'une seule fois.

**Implication pour le papier** : si Fast FixMatch est mentionné en annexe (comparaison/combinaison
avec EfficientMatch, RQ5), il faut préciser que son avantage théorique (FLOPs, indépendant du
hardware -- l'axe retenu par le protocole du papier, cf. §1) ne se traduit pas nécessairement par un
gain de temps d'horloge sur toute pile logicielle/matérielle, notamment quand `torch.compile` est
disponible pour les méthodes concurrentes mais pas pour lui.

## Bugs trouvés et corrigés pendant cette session de mesure

Trois problèmes indépendants ont dû être corrigés avant d'obtenir une mesure fiable :

1. **`data_root` relatif au répertoire de lancement** (`scripts/config.py`) : `"./data"` pointait vers
   un dossier différent selon que la commande était lancée depuis la racine du projet ou depuis
   `scripts/`, provoquant un retéléchargement complet de CIFAR-10 qui ressemblait à un blocage.
   Corrigé : chemin absolu calculé depuis l'emplacement du script, indépendant du cwd.
2. **Bug de fond, indépendant de la machine** (`scripts/data.py::infinite_loader`) : avec
   `n_labels=40` (ou toute valeur < `B=64`), le `DataLoader` labellisé (`shuffle=True`,
   `drop_last=True`) ne pouvait jamais former de batch complet -- la boucle infinie tournait dans le
   vide sans jamais rien produire (blocage silencieux, ~0% CPU/GPU). Corrigé : échantillonnage avec
   remise (`RandomSampler(replacement=True)`), pratique standard en SSL pour cycler sur un petit jeu
   de labels.
3. **Mesure Fast FixMatch faussée** (`scripts/benchmark_speed.py`) : fixer `cfg["K"]` à la taille de
   la fenêtre de mesure (au lieu du vrai budget cible) faisait balayer toute la courbe CBS en 40
   itérations, avec une taille de batch différente à presque chaque itération -- déclenchant une
   recherche cuDNN répétée qui donnait un résultat ~25× trop lent. Corrigé : mesure à taille de batch
   fixée à `u_t=cbs_min_batch` et à `u_t=mu*B`, puis intégration sur la vraie courbe CBS pour K=2¹⁷.

## Prochaine étape

Ces temps ne tiennent pas compte de l'arrêt anticipé (early stopping), qui devrait les réduire encore
-- à confirmer sur les runs pilotes. Voir la [feuille de route](https://claude.ai/code/artifact/88d24799-de53-4169-a124-01175a220300)
pour la suite (calibration du détecteur de plateau, ablation lambda_mix, puis Phase 3).
