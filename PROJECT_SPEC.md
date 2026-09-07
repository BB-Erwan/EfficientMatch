# EfficientMatch -- Spec du projet (handoff pour Claude Code)

## 1. Objectif du papier

Proposer **EfficientMatch**, une méthode de SSL (semi-supervised learning) en classification
d'images conçue pour une **convergence rapide à budget de calcul réduit**, plutôt que pour la
performance asymptotique visée par le protocole standard (`2^20` itérations). Le papier propose
aussi un **protocole d'évaluation réduit** (`2^17` itérations, early stopping, métriques
d'efficacité indépendantes du hardware) pour comparer les méthodes sous cet angle.

Cible : soumission ICLR 2027 (limite 9 pages corps + annexes illimitées).

## 2. Méthode : EfficientMatch

Structure FixMatch (perte supervisée + cohérence faible/forte filtrée par seuil de confiance)
**+ un canal de Mixup filtré** : Mixup appliqué entre le batch labellisé et le batch non
labellisé faiblement augmenté, où chaque paire mixée est pondérée par le **produit des masques
de confiance** de ses deux composants (masque dur ; un exemple étiqueté a un masque toujours = 1).

```
L_total = L_s (supervisée) + lambda_u * L_u (FixMatch classique) + lambda_mix * L_mix (Mixup filtré)
```

Différenciation par rapport aux baselines :
- vs. **MixMatch/ReMixMatch** : le Mixup n'utilise que des pseudo-labels validés par le seuil (pas de mélange non filtré).
- vs. **FixMatch/FlexMatch** : ajout d'un canal de régularisation supplémentaire, à coût de calcul marginal (pas de forward réseau additionnel autre que sur les données déjà mixées).

## 3. Méthodes comparées (baselines)

| Méthode | Mécanisme clé | Statut code |
|---|---|---|
| FixMatch | seuillage fixe (tau=0.95) + cohérence faible/forte | notebook fait |
| FlexMatch | seuillage adaptatif par classe (Curriculum Pseudo Labeling) | notebook fait |
| MixMatch | guessing (K augmentations) + sharpening + Mixup non filtré | notebook fait |
| Fast FixMatch | FixMatch + Curriculum Batch Size (CBS), formule B-EXP, alpha=0.7 | notebook fait |
| **EfficientMatch (ours)** | FixMatch + Mixup filtré (masque dur) | notebook fait |

Note : Fast FixMatch est limité au levier CBS seul (pas de CPL ni d'augmentation forte sur les
labels, les deux autres composants du papier original) -- simplification assumée et documentée
dans le papier comme limitation.

## 4. Protocole expérimental

- **Budget plafond** : `K = 2^17` itérations (au lieu de `2^20` standard), schedule cosine
  **recalé** sur ce budget : `eta = eta_0 * cos(7*pi*k / (16*K))`.
- **Early stopping** : détecteur de plateau basé sur la **pente d'une régression linéaire locale**
  de l'accuracy EMA (fenêtre glissante), pas un simple compteur de patience -- cf. `detect_plateau`
  dans `ssl_common.py`. Seuil et fenêtre à calibrer empiriquement sur des runs pilotes avant de
  figer les valeurs (actuellement en placeholder : fenêtre=5, seuil=1e-4).
  Les courbes des runs arrêtés tôt sont complétées par report de la dernière valeur EMA jusqu'à `K`
  pour permettre le calcul de l'AUC.
- **Datasets** : CIFAR-10 (cœur de l'étude), CIFAR-100 (perspective, non prioritaire), PathMNIST
  extrait de MedMNIST v2 (RQ3, non prioritaire, **loader pas encore implémenté**).
- **Budget de labels** : un seul budget par dataset, régime de faible labellisation (valeur exacte
  **non encore fixée** -- ex. 40 labels sur CIFAR-10, à trancher).
- **Architecture principale** : WideResNet-28-2. ResNet-18 en étude de robustesse secondaire
  (RQ4, non prioritaire).
- **Graines aléatoires** : 3 par configuration prioritaire (1 seule en premier passage si le temps
  presse, cf. section Priorités).
- **Hardware** : 1x GPU NVIDIA A4000 (16 Go), séquentiel (pas de parallélisation entre runs).

## 5. Métriques à logger à chaque évaluation (tous les `eval_every` itérations)

- accuracy (modèle EMA) en fonction de l'itération
- FLOPs cumulés (mesure réelle via `torch.utils.flop_counter.FlopCounterMode`, **une seule fois
  avant la boucle** puis multipliée/sommée -- ne jamais appeler `FlopCounterMode` à chaque itération,
  ça ralentit l'exécution)
- AUC de la courbe accuracy-vs-itérations, normalisée sur `[0, K]`
- itérations/FLOPs nécessaires pour atteindre un seuil de performance (seuil exact **non encore
  fixé** -- ex. 90% de l'accuracy asymptotique de FixMatch dans notre propre protocole, pas une
  valeur de la littérature à `2^20` itérations qui serait inatteignable à budget réduit)
- taux de masque (`mask_rate`) -- diagnostic, pas une métrique du papier

## 6. Hyperparamètres (déjà tranchés, cf. tableau dans le papier)

```
SGD, momentum=0.9, nesterov=True
lr initial = 0.03, schedule cosine recalé sur K=2^17
weight_decay = 5e-4 (CIFAR-10/PathMNIST) / 1e-3 (CIFAR-100)
B (batch labellisé) = 64, mu (ratio non labellisé) = 7
tau (seuil confiance) = 0.95
lambda_u = 1.0
EMA decay = 0.999
alpha_m (Mixup Beta) = 0.75
alpha_CBS (Fast FixMatch) = 0.7, cbs_min_batch = 8

NON ENCORE TRANCHÉS (ablation à faire en priorité 1, cf. section 7) :
lambda_mix (EfficientMatch) -- à choisir dans {0.5, 1, 2}
fenêtre + seuil du plateau detector -- à calibrer sur runs pilotes
n_labels exact par dataset
seuil de performance pour la métrique "itérations/FLOPs jusqu'à seuil"
```

## 7. Plan de priorité des expériences (contrainte : 1 seule A4000)

**Phase 1 (avant tout run long)** : mesurer it/s réel sur la config cible pour calibrer le
temps total attendu -- pas encore fait.

**Phase 2 (priorité 1)** : ablation `lambda_mix` sur EfficientMatch, budget réduit (`~2^14`),
1 seule seed, CIFAR-10, pour figer la valeur avant les runs complets.

**Phase 3 (priorité 2, cœur du papier -- RQ1+RQ2)** : FixMatch, FlexMatch, MixMatch,
EfficientMatch, sur CIFAR-10, budget de labels principal (faible labellisation), `K=2^17`,
au moins 1 seed pour commencer, puis compléter à 3 seeds si le temps le permet.

**Phase 4 (si le temps le permet, non bloquant pour la soumission résumé)** : PathMNIST (RQ3),
ResNet-18 (RQ4), Fast FixMatch + combinaison EfficientMatch+CBS (RQ5) -- destinés à l'Annexe du
papier, pas au corps principal.

## 8. Structure de fichiers attendue

```
ssl_common.py          -- FAIT : data, WideResNet, EMA, mesure FLOPs, detect_plateau, cosine_schedule
methods.py              -- À FAIRE : train_step_fixmatch, train_step_flexmatch, train_step_mixmatch,
                           train_step_fast_fixmatch, train_step_efficientmatch (une fonction par
                           méthode, PAS de superclasse -- chaque étape de l'algo doit rester explicite
                           dans le corps de la fonction, cf. notebooks existants pour le contenu exact
                           à reprendre/factoriser)
train.py                -- À FAIRE : script CLI (argparse) prenant --method, --dataset, --n_labels,
                           --seed, --K, etc., qui assemble ssl_common + methods et lance UN run complet,
                           avec logging JSON incrémental (itération, temps, FLOPs cumulés, accuracy,
                           pente du plateau detector) et arrêt anticipé intégré
run_priority_experiments.py -- À FAIRE : orchestrateur qui lance séquentiellement la file de la
                           section 7 (Phase 2 puis Phase 3) via subprocess sur train.py, avec reprise
                           possible (skip des runs déjà complétés en vérifiant l'existence et la
                           complétude du fichier de log JSON correspondant)
notebooks/              -- FAIT (5 notebooks .ipynb, un par méthode) : version pédagogique/exploratoire,
                           pas destinée à l'exécution automatisée en série -- à ne PAS confondre avec
                           train.py qui doit être le script de production pour run_priority_experiments.py
```

## 9. Ce que Claude Code doit vérifier/faire en priorité

1. Vérifier que `ssl_common.py` est cohérent et sans erreur (déjà écrit, à auditer).
2. Écrire `methods.py` en factorisant les 5 fonctions `train_step_*` à partir du contenu des
   notebooks existants (`*_experiment.ipynb`) -- garder l'esprit "pas de superclasse, chaque ligne
   explicite" déjà appliqué dans les notebooks.
3. Écrire `train.py` (CLI) et `run_priority_experiments.py` (orchestrateur) selon la section 8.
4. Implémenter le loader PathMNIST (package `medmnist`) dans `ssl_common.py::load_datasets`,
   actuellement un stub qui lève `NotImplementedError`.
5. Trancher ou signaler explicitement les valeurs encore en placeholder listées en section 6.
6. Vérifier la cohérence de la formule Curriculum Batch Size (B-EXP, alpha=0.7) reprise du papier
   Fast FixMatch (Chen, Dun & Kyrillidis, arXiv:2309.03469) contre l'implémentation actuelle avant
   de lancer un run long dessus -- c'est la partie la moins vérifiée du code à ce stade.
7. Ne pas lancer de run complet sans avoir d'abord fait le test de mesure it/s (Phase 1, section 7).
