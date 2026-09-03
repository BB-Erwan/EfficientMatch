# EfficientMatch

Étude comparative d'algorithmes d'apprentissage semi-supervisé (SSL) sur CIFAR-10/100, en régime
faible labellisation (`n_labels` réduit, protocole budget `2**17` itérations). Le dépôt contient
deux façons équivalentes de lancer les mêmes expériences :

- des **notebooks** (`notebooks/`), un par algorithme, pour l'exploration interactive ;
- un **framework en ligne de commande** (`scripts/`), qui atomise la même logique en modules `.py`
  pour lancer des runs longs/reproductibles avec tous les hyperparamètres configurables.

## Algorithmes implémentés

| Algorithme | Notebook | Référence | Idée clé |
|---|---|---|---|
| FixMatch | [notebooks/fixmatch_experiment.ipynb](notebooks/fixmatch_experiment.ipynb) | Sohn et al., 2020 | Baseline : perte supervisée + cohérence faible/forte filtrée par un seuil de confiance fixe (`tau`). |
| FlexMatch | [notebooks/flexmatch_experiment.ipynb](notebooks/flexmatch_experiment.ipynb) | Zhang et al., 2021 | FixMatch + seuillage **adaptatif par classe** (Curriculum Pseudo Labeling), pour corriger le biais envers les classes "faciles". |
| MixMatch | [notebooks/mixmatch_experiment.ipynb](notebooks/mixmatch_experiment.ipynb) | Berthelot et al., 2019 | Pas d'augmentation forte : guessing par moyenne de K augmentations faibles + sharpening + MixUp entre labellisé et non labellisé. |
| Fast FixMatch | [notebooks/fast_fixmatch_experiment.ipynb](notebooks/fast_fixmatch_experiment.ipynb) | Chen, Dun & Kyrillidis, 2023/2024 | FixMatch + **Curriculum Batch Size** : la taille du batch non labellisé croît progressivement au cours de l'entraînement pour accélérer le début du training. |
| EfficientMatch | [notebooks/efficientmatch_experiment.ipynb](notebooks/efficientmatch_experiment.ipynb) | (contribution de ce dépôt) | FixMatch + canal de **Mixup filtré** par le masque de confiance dur, entre le batch labellisé et le batch non labellisé faiblement augmenté. |

Les 5 notebooks partagent une structure strictement identique (imports, config, données, modèle
WideResNet, EMA/FLOPs, assemblage) : seule la section 6 (`train_step_<algo>`) diffère, pour isoler
précisément l'effet de chaque contribution algorithmique sur les résultats.

## Structure du dépôt

```
notebooks/
    fixmatch_experiment.ipynb
    flexmatch_experiment.ipynb
    mixmatch_experiment.ipynb
    fast_fixmatch_experiment.ipynb
    efficientmatch_experiment.ipynb
scripts/
    train.py            # point d'entrée CLI
    config.py           # config par défaut (commune + spécifique à chaque algo)
    data.py             # datasets CIFAR SSL + transforms v1/v2
    models.py           # WideResNet
    ema.py              # EMA des poids
    schedule.py         # schedule de learning rate cosine recalé
    evaluate.py         # évaluation top-1
    engine.py           # boucle d'entraînement générique (partagée par tous les algos)
    algorithms/         # train_step_<algo> + calcul des FLOPs, un module par algorithme
    requirements.txt
    README.md           # détails d'utilisation du framework CLI
```

## Utilisation

### Notebooks

Ouvrir le notebook de l'algorithme voulu dans `notebooks/` et exécuter les cellules dans l'ordre ;
chaque notebook télécharge CIFAR-10/100 dans `./data` et écrit ses logs dans `./logs_<algo>.json`.

### Ligne de commande

```powershell
pip install -r scripts/requirements.txt

python scripts/train.py --algo fixmatch
python scripts/train.py --algo efficientmatch --n-labels 250 --K 65536 --no-use-amp
python scripts/train.py --algo mixmatch --set weight_decay=1e-3 --set rampup_length=8000
```

Tous les hyperparamètres (dataset, budget d'entraînement, optimisations de vitesse, hyperparamètres
propres à chaque algorithme...) sont exposés en flags CLI. Voir [scripts/README.md](scripts/README.md)
pour le détail complet des options et des exemples.

## Métriques suivies

Chaque run (notebook ou CLI) journalise, à intervalles réguliers (`eval_every`), l'accuracy top-1 sur
le jeu de test (poids EMA), la perte, les taux de masquage des pseudo-étiquettes et les FLOPs cumulés
(mesurés via `torch.utils.flop_counter.FlopCounterMode`), dans un fichier `logs_<algo>.json`.

## Licence

[MIT](LICENSE)

