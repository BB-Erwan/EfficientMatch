# Scripts CLI -- EfficientMatch

Framework `.py` (hors notebooks) qui atomise les étapes des notebooks `notebooks/*.ipynb` en modules
réutilisables, pour lancer les mêmes expériences SSL en ligne de commande avec des hyperparamètres
entièrement configurables. Pour l'instant, seul FixMatch est implémenté.

## Structure

```
scripts/
    train.py        # point d'entrée CLI (un seul run)
    analyze.py       # post-traitement des logs JSON -> AUC, itérations jusqu'à seuil, forward-fill
    config.py        # config par défaut (commune + spécifique à chaque algo + par dataset)
    data.py          # datasets CIFAR-10/100/PathMNIST SSL + transforms
    models.py        # WideResNet-28-2
    ema.py           # EMA des poids
    schedule.py      # schedule de learning rate cosine recalé
    evaluate.py       # évaluation top-1
    engine.py        # boucle d'entraînement générique
    algorithms/
        fixmatch.py
```

Chaque module de `algorithms/` expose `make_train_step(cfg, device)`. `engine.py` assemble le reste
(données, modèle, optimiseur, logging JSON) et appelle cette fonction -- exactement la logique des
notebooks, mais atomisée en fichiers indépendants.

## Installation

```powershell
pip install -r scripts/requirements.txt
```

## Utilisation

```powershell
# Lancer FixMatch avec les hyperparamètres par défaut (identiques au notebook)
python scripts/train.py --algo fixmatch

# Changer des hyperparamètres exposés explicitement (n'importe quelle clé de CONFIG)
python scripts/train.py --algo fixmatch --n-labels 250 --K 65536 --tau 0.9

# Surcharger un paramètre non exposé explicitement (répétable), valeur interprétée via ast.literal_eval
python scripts/train.py --algo fixmatch --set weight_decay=1e-3

# Sous-ensemble de debug pour vérifier rapidement le pipeline
python scripts/train.py --algo fixmatch --debug-subset-size 2000 --K 200 --eval-every 50
```

Toutes les clés de `CONFIG` (voir `config.py`) sont exposées en flags `--nom-de-cle` (underscores ->
tirets). Les booléens utilisent `--flag`/`--no-flag`. `--dataset` bascule automatiquement
`num_classes`/`weight_decay` selon le dataset (cf. `config.DATASET_DEFAULTS`), sauf si vous les
surchargez vous-même explicitement.

Chaque run écrit sa configuration + ses logs (perte, accuracy) dans
`./logs/<algo>_<dataset>_n<n_labels>_K<K>_seed<seed>[_<tag>].json` (le `tag` optionnel, via `--tag`,
sert à distinguer plusieurs runs qui partagent (algo, dataset, n_labels, K, seed)). Le fichier est
réécrit à chaque évaluation puis, à la fin du run (budget atteint ou arrêt anticipé), marqué
`"status": "completed"`.

## Analyse des résultats (métriques du papier, absentes des logs bruts)

`train.py`/`engine.py` ne logguent que des points bruts (itération, accuracy). `analyze.py` calcule
après coup : AUC normalisée sur `[0, K]`, report de la dernière valeur EMA jusqu'à `K` pour les runs
arrêtés tôt (pour que l'AUC reste comparable entre méthodes), et itérations pour atteindre une
fraction de l'accuracy asymptotique d'un algo de référence (FixMatch par défaut).

```powershell
python scripts/analyze.py --logs-dir ./logs --dataset cifar10 --n-labels 250 --K 131072 \
    --reference-algo fixmatch --threshold-frac 0.9 --out results.json
```
