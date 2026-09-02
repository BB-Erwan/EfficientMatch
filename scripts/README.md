# Scripts CLI -- EfficientMatch

Framework `.py` (hors notebooks) qui atomise les étapes des notebooks `notebooks/*.ipynb` en modules
réutilisables, pour lancer les mêmes expériences SSL en ligne de commande avec des hyperparamètres
entièrement configurables.

## Structure

```
scripts/
    train.py        # point d'entrée CLI
    config.py        # config par défaut (commune + spécifique à chaque algo)
    data.py          # datasets CIFAR SSL + transforms v1/v2 (bascule CONFIG["use_transforms_v2"])
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

Chaque module de `algorithms/` expose deux fonctions : `estimate_flops_per_iter(model, cfg, device)` et
`make_train_step(cfg, augmenter, weak_transform, strong_transform, device)`. `engine.py` assemble le
reste (données, modèle, optimiseur, logging JSON) et appelle ces fonctions -- exactement la logique des
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
-- lancez `python scripts/train.py --algo <nom> --help` pour voir la liste complète.

Chaque run écrit sa configuration + ses logs (perte, accuracy, FLOPs cumulés) dans
`./logs_<algo>.json`, au même format que les notebooks.
