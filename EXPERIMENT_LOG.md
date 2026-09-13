# Journal d'expériences — EfficientMatch (session Claude Code)

Ce document résume tout ce qui a été fait en session Claude Code depuis la spec initiale
(`PROJECT_SPEC.md`) : changements de code, expériences lancées, résultats obtenus, et problèmes
identifiés. Objectif : servir de brief à un futur assistant (Claude ou autre) pour aider à la
rédaction de la publication, sans avoir à rejouer toute l'historique de la session.

Pour le contexte méthodologique complet (objectif du papier, formule d'EfficientMatch, protocole,
hyperparamètres tranchés), voir `PROJECT_SPEC.md`. Ce document-ci ne répète pas ce qui y est déjà
écrit — il documente ce qui s'est passé **après**.

---

## 1. Changements de code apportés en session

### 1.1 Nouvelle variante `efficientmatch_2.py`

Une deuxième implémentation d'EfficientMatch a été créée (`scripts/efficientmatch_2.py`), quasi
identique à `efficientmatch.py` sauf sur le calcul de la perte Mixup :

- **efficientmatch (v1)** : deux `cross_entropy` séparées, une sur la portion labellisée du batch
  mixé, une sur la portion non labellisée, chacune pondérée par son masque correspondant.
- **efficientmatch_2** : une seule `cross_entropy` sur le batch mixé complet (labellisé +
  non labellisé concaténés), pondérée par le masque complet — fusion des deux forward en un seul
  calcul de perte. Autre différence : `mixup_targets_u` utilise le pseudo-label dur
  (`F.one_hot(pseudo, ...)`) plutôt que la distribution de probabilité molle `probs_u_w`.

**Effet observé** : efficientmatch_2 est systématiquement aussi rapide ou plus rapide que
efficientmatch (v1) pour atteindre un seuil de performance donné (voir §3). Sur CIFAR-10 250
labels/seed 2312, seuil 80% : efficientmatch_2 en 33.4 min contre 59.0 min pour la v1 (~1.8x plus
rapide). Cause probable : le calcul fusionné réduit l'overhead de deux `cross_entropy` séparées, et
l'usage du pseudo-label dur plutôt que la distribution molle pourrait donner un signal
d'apprentissage plus net sur les pseudo-labels. **Non isolé/ablationné** — les deux changements
(fusion de perte + pseudo-label dur) ont été faits ensemble, donc on ne sait pas lequel des deux
explique le gain. À creuser si le papier veut attribuer le gain à un mécanisme précis.

### 1.2 Nouveau script `supervised.py`

Baseline pleinement supervisée (tout le jeu d'entraînement labellisé, pas de split labeled/
unlabeled, pas de pseudo-labeling, `cross_entropy` standard) pour mesurer le plafond de
performance brut d'un backbone. Utile pour situer où se trouve la performance atteignable avec
tous les labels, comme référence haute pour les courbes SSL. Non encore exécuté jusqu'au bout dans
cette session (un premier essai DenseNet/CIFAR-100 a été interrompu, voir §4.3).

### 1.3 Nouveau backbone : ResNet-18 adapté CIFAR

Ajouté par l'utilisateur directement dans `models.py` (`resnet18_cifar`) : `torchvision.models
.resnet18` avec `conv1` remplacé par un 3×3 stride 1 (au lieu du 7×7 stride 2 standard ImageNet)
et `maxpool` supprimé (`nn.Identity()`), adaptation classique pour des images 32×32. Intégré à la
factory commune `build_model(model_name, num_classes, widen_factor, depth)` dans `models.py`,
elle-même appelée par les 7 scripts d'expérience via un nouvel argument `--model
{wideresnet,resnet18}` (défaut `wideresnet` pour compat, sauf `supervised.py` où le défaut est
`resnet18`). ResNet-18 CIFAR : 11 220 132 paramètres.

Un DenseNet-BC-100-12 (Huang et al. 2016, ~0.8M paramètres) a aussi été implémenté à la demande de
l'utilisateur puis **retiré des choix `--model`** sur instruction explicite ("oublie le densenet")
au profit du ResNet-18. Le code de `DenseNetCIFAR` reste présent dans `models.py` mais n'est plus
référencé par aucun script.

### 1.4 WideResNet profondeur variable (`--depth`)

`build_model` et les 7 scripts acceptent maintenant `--depth` (défaut 28) pour construire des
WideResNet-40-x en plus des WRN-28-x historiques (contrainte `(depth-4) mod 6 == 0`). Testé :
WRN-40-2 → 2 255 156 paramètres, sortie correcte.

### 1.5 Accuracy top-k

`utils.py::evaluate_f1_and_accuracy` accepte maintenant `topk` (défaut 1) et retourne
`(f1, acc, topk_accs)` — `topk_accs` est une liste de longueur `topk` où l'indice `k-1` est
l'accuracy top-k, calculée en une seule passe sur le test set (pas de coût de calcul
supplémentaire notable). Les 7 scripts ont un argument `--topk` ; quand `--topk > 1`, les logs
affichent le détail `Top-1: x, Top-2: y, Top-3: z, ...` et le JSON de métriques gagne une clé
`"topk_acc"` (liste de listes, une par évaluation). Comportement inchangé quand `--topk=1`
(défaut).

### 1.6 Correction de la convention de nommage des dossiers de résultats

Bug découvert et corrigé : `dataset_prefix` était vide pour `cifar10` (`f"{args.dataset}-" if
args.dataset != "cifar10" else ""`) dans les 6 scripts originaux, donc un run CIFAR-10 écrivait
dans `results/labeled-{n}-seed-{s}/` alors que tous les runs historiques CIFAR-10 vivaient dans
`results/cifar10-labeled-{n}-seed-{s}/` (dossier créé à la main ou par une version antérieure du
code). Corrigé pour que le préfixe du dataset soit **toujours** inclus, dans les 6 scripts +
`efficientmatch_2.py` + `supervised.py`. Les résultats déjà produits sous l'ancien schéma
(`results/labeled-250-seed-2312/`, `results/labeled-250-seed-308/`) ont été déplacés à la main
vers `results/cifar10-labeled-250-seed-*/`.

Autre migration : à un moment de la session, tous les résultats sont passés de
`scripts/results/` à `results/` (racine du repo) — `results_dir` est maintenant calculé par
rapport à la racine du repo (`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`)
plutôt que relatif au répertoire d'exécution courant.

### 1.7 Support `--dataset`, `--widen_factor`, `--weight_decay` (avant cette session)

D'après l'historique git (commit `509f15d`), fixmatch/mixmatch/flexmatch/sequencematch avaient
déjà reçu le support de `--dataset {cifar10,cifar100,svhn}`, `--widen_factor` et `--weight_decay`
avant le début de cette session — mentionné ici pour contexte, pas un changement de cette session.

---

## 2. Environnement d'exécution

- **GPU** : NVIDIA GeForce **RTX 5060 Ti** (8151 MiB VRAM) — à noter que ceci **diffère** du GPU
  documenté dans `BENCHMARK_RESULTS.md` (RTX A4000, 16 Go) et dans `PROJECT_SPEC.md` §4 (A4000
  également) : les temps de cette session ne sont donc pas directement comparables aux mesures de
  `BENCHMARK_RESULTS.md` sans normalisation.
- Sur cette RTX 5060 Ti, les scripts activent `torch.compile(mode="reduce-overhead")` (condition
  codée en dur sur `"5060 Ti" in torch.cuda.get_device_name(0)`), ce qui impose un **coût de
  compilation JIT ponctuel** de plusieurs minutes en début de run avant que la vitesse de croisière
  ne soit atteinte (observé sur efficientmatch_2 et sur le premier run de `supervised.py`).
- OS Windows, exécution via `conda run -n DEEP-GPU python <script>.py`.

---

## 3. Résultats des expériences SSL (SVHN, 250 labels/classe budget)

Sweep principal : **efficientmatch_2, fixmatch, flexmatch, mixmatch** sur **SVHN, 250 labels**,
seeds **2312, 0308, 2701**, `target_acc=0.90` (arrêt anticipé dès que l'accuracy EMA atteint 90%).
`efficientmatch` (v1) a été testé une fois sur seed 2312 (90.05%, 6.3 min) puis volontairement
exclu des seeds suivantes sur demande utilisateur.

### 3.1 Steps/temps/FLOPs pour atteindre 90% (SVHN, 250 labels)

FLOPs/itération mesurés via `scripts/flops_analysis.py` (indépendants du device) :
fixmatch/flexmatch = 850.10 GFLOPs, mixmatch = 301.64 GFLOPs, efficientmatch(v1)/efficientmatch_2
≈ 740.35 GFLOPs (efficientmatch_2 non mesuré séparément par le script, valeur approximée par
celle d'efficientmatch v1 — mêmes formes de tenseurs).

| Seed | Méthode | Steps | Temps | FLOPs totaux |
|---|---|---:|---:|---:|
| 2312 | efficientmatch_2 | 8500 | 405.7s (6.76 min) | 6 293.0 TFLOPs |
| 2312 | fixmatch | 8000 | 469.4s (7.82 min) | 6 800.8 TFLOPs |
| 2312 | mixmatch | 18500 | 408.8s (6.81 min) | **5 580.3 TFLOPs** |
| 2312 | flexmatch | — | ❌ jamais atteint (max 15.96%, diverge) | — |
| 0308 | efficientmatch_2 | 8500 | 405.5s (6.76 min) | 6 293.0 TFLOPs |
| 0308 | fixmatch | 6500 | **386.1s (6.44 min)** | **5 525.7 TFLOPs** |
| 0308 | mixmatch | 52500 | 1103.0s (18.38 min) | 15 836.1 TFLOPs |
| 0308 | flexmatch | — | ❌ jamais atteint (max 82.38%, plateau) | — |
| 2701 | efficientmatch_2 | 10000 | 471.4s (7.86 min) | 7 403.5 TFLOPs |
| 2701 | fixmatch | 8000 | 473.8s (7.90 min) | **6 800.8 TFLOPs** |
| 2701 | mixmatch | 35000 | 746.5s (12.44 min) | 10 557.4 TFLOPs |
| 2701 | flexmatch | ~40000 | ❌ arrêté à 37.99 min (max 85.10%, jamais 90%) | ~34 000 TFLOPs (non atteint) |

**Pas de vainqueur unique et cohérent d'une seed à l'autre** :
- En **temps réel**, efficientmatch_2 et fixmatch sont systématiquement les deux plus rapides
  (quasi ex æquo sur 2312 et 2701), mixmatch nettement plus lent sauf sur seed 2312.
- En **FLOPs**, le classement s'inverse selon la seed : mixmatch gagne sur seed 2312 (moins de
  steps que d'habitude pour cette seed) mais devient le plus coûteux sur 0308 et 2701 (besoin de
  beaucoup plus de steps, 52500 et 35000) malgré son coût par itération très faible.
- **fixmatch est la valeur la plus sûre** : toujours dans le top 2 sur les deux métriques, sur les
  3 seeds.
- **flexmatch échoue systématiquement** — voir §5.

### 3.2 Temps jusqu'à 85% (seuil intermédiaire, seed 2312 uniquement)

| Méthode | Step | Temps | Acc |
|---|---:|---:|---:|
| fixmatch | 4500 | 4.9 min | 85.34% |
| mixmatch | 9000 | 3.6 min | 85.52% |
| efficientmatch (v1) | 7500 | 6.3 min | 85.59% |
| efficientmatch_2 | 5500 | **4.7 min** | 85.69% |
| flexmatch | — | jamais atteint | — |

efficientmatch_2 franchit 85% plus vite qu'efficientmatch v1 (4.7 vs 6.3 min) et talonne fixmatch,
qui reste la référence de vitesse à ce seuil intermédiaire aussi.

---

## 4. Résultats sur CIFAR-10 (250 labels)

### 4.1 Comparaison à seuil 80% (seed 2312)

Table historique complète (avant le début de cette session, mais réutilisée comme référence pour
comparer efficientmatch_2) :

| Méthode | Step | Temps |
|---|---:|---:|
| **efficientmatch_2** | 45000 | **33.4 min** |
| efficientmatch_flex | 47000 | 34.8 min |
| efficientmatch_flex_mu2 | 88000 | 50.7 min |
| efficientmatch (v1) | 82000 | 59.0 min |
| flexmatch | 70500 | 114.1 min |
| fixmatch | 170500 | 272.5 min |
| mixmatch | — | jamais atteint 80% (max 78.42% après **13 heures**, 46780s, 1658 évaluations) |

efficientmatch_2 est la méthode la plus rapide sur cette seed, ~1.8x plus rapide que efficientmatch
v1 et ~8x plus rapide que fixmatch.

### 4.2 efficientmatch_2 sur seeds supplémentaires (CIFAR-10, seuil 80%)

- **Seed 0308** : 90.35%(sic, en réalité seuil visé 80%, atteint) — atteint en **40.2 min**
  (2409.6s), plus lent que sur seed 2312 (33.4 min). Aucune donnée comparative des autres
  méthodes sur cette seed précise.
- **Seed 2701** : atteint 80.35% au step 34000 (première tentative), avant qu'un incident de
  lancement double (voir §6.2) n'oblige à supprimer et relancer ; le relancement final a atteint
  80.00% au step 34000 également.
- **Seed 666** : lancée par erreur en double avec seed 2701 (incident §6.2), résultat supprimé
  sans avoir été exploité.

**Anomalie de dossier découverte puis corrigée** : les runs `efficientmatch_2` sur CIFAR-10
écrivaient dans `results/labeled-250-seed-*/` (sans préfixe `cifar10-`) à cause du bug §1.6,
alors que toutes les données historiques des autres méthodes vivent dans
`results/cifar10-labeled-250-seed-*/`. Les fichiers ont été déplacés manuellement seed par seed
au fur et à mesure ; le bug est corrigé pour tout run futur.

### 4.3 CIFAR-10 seed 2701, target 80% (sweep en cours à la fin de la session)

Sweep lancé après vérification qu'aucun résultat n'existait déjà pour cette combinaison :
`efficientmatch_2, fixmatch, flexmatch, mixmatch`, `--dataset cifar10 --num_labeled 250 --seed
2701 --target_acc 0.80`.

- **efficientmatch_2** : atteint 80.00% au step 34000.
- **fixmatch** : en cours au moment de la clôture de ce journal (58.50% après 16 évaluations,
  7.14 min) — **résultat final non disponible**, à vérifier dans
  `results/cifar10-labeled-250-seed-2701/fixmatch_ema_metrics.json` à la reprise.
- **flexmatch, mixmatch** : pas encore démarrés au moment de la clôture de ce journal.

---

## 5. Problème identifié : flexmatch échoue systématiquement sur SVHN/250 labels/target 90%

Sur les **3 seeds testées** (2312, 0308, 2701), flexmatch n'a **jamais atteint 90%** d'accuracy
sur SVHN à 250 labels, plafonnant entre **82% et 85%** selon la seed, avec des temps d'exécution
5 à 12x supérieurs aux autres méthodes pour un score inférieur. Détail complet (steps, temps,
diagnostic par seed) dans `RUNS_TO_REVISIT.md`. Sur la seed 2312 spécifiquement, un premier
essai a carrément **divergé** (max 15.96%) et a dû être interrompu manuellement.

**Ce n'est vraisemblablement pas de la variance aléatoire** — la reproductibilité du plafond sur
3 seeds indépendantes pointe vers un problème méthodologique : hyperparamètres flexmatch (seuil
adaptatif par classe / `thresh_warmup`) potentiellement mal calibrés pour le régime SVHN à 250
labels, ou une interaction avec le learning rate/schedule. **Non résolu** — à investiguer avant de
citer flexmatch comme baseline fiable dans le papier sur cette configuration précise. Les runs sur
CIFAR-10 (250 labels) n'ont pas montré le même échec catégorique (flexmatch y atteint 80% en
114 min, plus lent que les autres mais pas bloqué en dessous du seuil) — le problème semble donc
spécifique à SVHN/250 labels/target 90%, pas une régression générale de l'implémentation.

---

## 6. Incidents opérationnels de la session (pour info, pas des résultats scientifiques)

### 6.1 CIFAR-100 250 et 2500 labels (fixmatch)

Deux runs fixmatch lancés en tout début de session sur CIFAR-100 (seed 2312) :
- **250 labels** : très faible accuracy (5-7% après 90 époques, `target_acc=0.60` jamais
  approché) — attendu vu le régime extrême (2.5 labels/classe sur 100 classes). Run finalement
  arrêté par l'utilisateur.
- **2500 labels** : bien meilleur, ~41% d'accuracy après 52 époques avant d'être arrêté par
  l'utilisateur pour lancer d'autres priorités. N'a pas été relancé jusqu'à complétion dans cette
  session.

### 6.2 Double lancement accidentel (efficientmatch_2, CIFAR-10 seeds 2701/666)

Un script d'attente basé sur `pgrep -f` (pour enchaîner un nouveau run seulement après la fin
d'un précédent) a échoué à détecter des process encore actifs lancés depuis une autre session
bash — `pgrep` sous Git Bash sur Windows ne voit pas fiablement les processus d'un autre arbre de
process. Résultat : deux runs `efficientmatch_2` ont tourné en parallèle sur le même GPU (seeds
2701 et 666), dégradant les deux. Les deux ont été arrêtés et leurs résultats supprimés sur
demande utilisateur. **Leçon opérationnelle** : ne plus utiliser `pgrep`/scripts d'attente pour
séquencer des lancements — préférer attendre la notification de fin de tâche du harness avant de
lancer la suite.

### 6.3 `supervised.py` (ResNet-18/DenseNet, CIFAR-100) — lenteur anormale, cause non confirmée

Un premier essai de `supervised.py` (DenseNet, CIFAR-100 complet) a montré une lenteur
**soutenue** (pas seulement un warmup ponctuel de `torch.compile`) : ~310s par période de 500
steps, constant sur 2 évaluations consécutives. Hypothèse avancée : dernier batch de chaque epoch
de taille irrégulière (50000 images / batch_size 128 ⇒ dernier batch de 80) forçant une
recompilation CUDA-graph (`torch.compile(mode="reduce-overhead")`) à chaque epoch. **Hypothèse
non confirmée expérimentalement** — le run a été arrêté sur décision utilisateur avant qu'un test
avec `drop_last=True` ait pu être fait, et l'utilisateur a ensuite demandé explicitement de **ne
pas** ajouter `drop_last`. Le script utilise maintenant ResNet-18 par défaut (plus DenseNet) ;
l'anomalie de vitesse n'a pas été revérifiée avec ce nouveau backbone.

---

## 7. État des lieux / ce qu'il reste à faire (pour la suite)

- **flexmatch sur SVHN** (§5) : investiguer la cause du plafond, ou documenter le résultat tel
  quel comme une limite observée de la méthode dans ce régime.
- **CIFAR-10 seed 2701** (§4.3) : sweep incomplet à la clôture de la session (fixmatch en cours,
  flexmatch/mixmatch pas lancés) — à terminer.
- **CIFAR-100** (§6.1) : aucun run mené à terme jusqu'à présent dans cette session — le budget de
  labels exact (§6 de `PROJECT_SPEC.md`) et les seuils cibles restent à fixer pour ce dataset.
- **`supervised.py`** (§6.3, §1.2) : script prêt mais jamais exécuté jusqu'au bout ; utile comme
  référence de plafond de performance (accuracy avec 100% des labels) à citer dans le papier en
  face des courbes SSL à faible budget de labels.
- **Ablation efficientmatch_2 vs v1** (§1.1) : le gain de vitesse d'efficientmatch_2 mélange deux
  changements (fusion de perte + pseudo-label dur) jamais isolés séparément — à faire si le papier
  veut présenter efficientmatch_2 comme une amélioration justifiée mécanistiquement plutôt qu'une
  variante empirique.
- **Écart de matériel** (§2) : les mesures de temps de cette session (RTX 5060 Ti) ne sont pas
  directement comparables à `BENCHMARK_RESULTS.md` (RTX A4000) ni au hardware visé par
  `PROJECT_SPEC.md` §4 — les FLOPs restent la métrique comparable entre les deux, le temps
  d'horloge non.
- **`--topk`, `--depth`, `--model resnet18`** (§1.3-1.5) : fonctionnalités ajoutées et testées
  unitairement (sanity checks passés) mais **aucune expérience réelle ne les a encore utilisées** —
  aucun résultat top-k, WRN-40-x, ou ResNet-18 n'existe encore dans `results/` à ce jour.
