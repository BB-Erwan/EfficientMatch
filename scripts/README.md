# Scripts CLI -- EfficientMatch

Framework `.py` (hors notebooks) qui atomise les étapes des notebooks `notebooks/*.ipynb` en modules
réutilisables, pour lancer les mêmes expériences SSL en ligne de commande avec des hyperparamètres
entièrement configurables.

## Structure

```
scripts/
    train.py        # point d'entrée CLI (un seul run)
    run_priority_experiments.py  # orchestrateur séquentiel Phase 2 (ablation lambda_mix) + Phase 3
    analyze.py       # post-traitement des logs JSON -> AUC, itérations/FLOPs jusqu'à seuil, forward-fill
    config.py        # config par défaut (commune + spécifique à chaque algo + par dataset)
    data.py          # datasets CIFAR-10/100/PathMNIST SSL + transforms v1/v2
    models.py        # WideResNet-28-2
    ema.py           # EMA des poids
    schedule.py      # schedule de learning rate cosine recalé
    evaluate.py       # évaluation top-1
    engine.py        # boucle d'entraînement générique (partagée par tous les algos)
    algorithms/
        fixmatch.py
        fast_fixmatch.py
        flexmatch.py
        mixmatch.py
        efficientmatch.py
```

Chaque module de `algorithms/` expose trois fonctions : `estimate_flops_per_iter(model, cfg, device)`,
`flops_for_step(flops_measurement, step_metrics)` (traduit la mesure en FLOPs réels de l'itération
courante -- constant pour la plupart des algos, mais dépend de `step_metrics["u_t"]` pour Fast FixMatch
dont la taille de batch varie au cours du run) et `make_train_step(cfg, augmenter, weak_transform,
strong_transform, device)`. `engine.py` assemble le reste (données, modèle, optimiseur, logging JSON)
et appelle ces fonctions -- exactement la logique des notebooks, mais atomisée en fichiers indépendants.

## Installation

```powershell
pip install -r scripts/requirements.txt
```

## Utilisation

```powershell
# Lancer FixMatch avec les hyperparamètres par défaut (identiques au notebook)
python scripts/train.py --algo fixmatch

# Changer des hyperparamètres exposés explicitement (n'importe quelle clé de CONFIG)
python scripts/train.py --algo efficientmatch --n-labels 250 --K 65536 --no-use-amp --tau 0.9

# Revenir aux transforms v1 (classique, par image) au lieu de v2 (batch vectorisé)
python scripts/train.py --algo flexmatch --no-use-transforms-v2

# Surcharger un paramètre non exposé explicitement (répétable), valeur interprétée via ast.literal_eval
python scripts/train.py --algo mixmatch --set weight_decay=1e-3 --set rampup_length=8000

# Sous-ensemble de debug pour vérifier rapidement le pipeline
python scripts/train.py --algo fast_fixmatch --debug-subset-size 2000 --K 200 --eval-every 50
```

Toutes les clés de `CONFIG` (voir `config.py`) sont exposées en flags `--nom-de-cle` (underscores ->
tirets). Les booléens utilisent `--flag`/`--no-flag`. Les paramètres spécifiques à un algorithme
(`alpha_mix`, `K_aug`, `cbs_alpha`, ...) n'apparaissent que lorsque `--algo` correspondant est sélectionné
-- lancez `python scripts/train.py --algo <nom> --help` pour voir la liste complète. `--dataset` bascule
automatiquement `num_classes`/`weight_decay` selon le dataset (cf. `config.DATASET_DEFAULTS`), sauf si
vous les surchargez vous-même explicitement.

Chaque run écrit sa configuration + ses logs (perte, accuracy, FLOPs cumulés) dans
`./logs/<algo>_<dataset>_n<n_labels>_K<K>_seed<seed>[_<tag>].json` (le `tag` optionnel, via `--tag`,
sert à distinguer plusieurs runs qui partagent (algo, dataset, n_labels, K, seed) -- ex. l'ablation
lambda_mix). Le fichier est réécrit à chaque évaluation puis, à la fin du run (budget atteint ou arrêt
anticipé), marqué `"status": "completed"` -- c'est ce marqueur que `run_priority_experiments.py` utilise
pour sauter les runs déjà complétés à la reprise.

## Orchestrateur (Phase 2 + Phase 3, PROJECT_SPEC.md §7)

```powershell
# Aperçu de la file complète sans rien lancer
python scripts/run_priority_experiments.py --lambda-mix-frozen 1.0 --dry-run

# Phase 2 seule : ablation lambda_mix (EfficientMatch, budget réduit K=2**14, 1 seed, {0.5, 1, 2})
python scripts/run_priority_experiments.py --skip-phase3

# Phase 3 seule, une fois lambda_mix figé (cf. analyze.py ci-dessous), 3 graines
python scripts/run_priority_experiments.py --skip-phase2 --lambda-mix-frozen 1.0 --phase3-seeds 0 1 2
```

Un run déjà présent dans `./logs/` avec `"status": "completed"` est automatiquement sauté -- on peut
donc interrompre et relancer l'orchestrateur sans dupliquer de travail.

## Analyse des résultats (métriques du papier, absentes des logs bruts)

`train.py`/`engine.py` ne logguent que des points bruts (itération, accuracy, FLOPs cumulés).
`analyze.py` calcule après coup ce que le papier rapporte réellement (Table 1) : AUC normalisée sur
`[0, K]`, report de la dernière valeur EMA jusqu'à `K` pour les runs arrêtés tôt (pour que l'AUC reste
comparable entre méthodes), et itérations/FLOPs pour atteindre une fraction de l'accuracy asymptotique
d'un algo de référence (FixMatch par défaut).

```powershell
python scripts/analyze.py --logs-dir ./logs --dataset cifar10 --n-labels 40 --K 131072 \
    --reference-algo fixmatch --threshold-frac 0.9 --out results_phase3.json
```
