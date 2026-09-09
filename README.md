# EfficientMatch

Étude comparative d'algorithmes d'apprentissage semi-supervisé (SSL) sur CIFAR-10, en régime faible
labellisation (`n_labels` réduit). Chaque algorithme est un **script autonome** dans `scripts/` :
augmentations, hyperparamètres et boucle d'entraînement vivent tous dans le même fichier (rien n'est
factorisé entre algorithmes sauf ce qui est strictement identique quel que soit l'algorithme --
architecture du modèle, EMA, évaluation).

## Algorithmes implémentés

| Algorithme | Script | Référence | Idée clé |
|---|---|---|---|
| FixMatch | [scripts/fixmatch.py](scripts/fixmatch.py) | Sohn et al., 2020 | Baseline : perte supervisée + cohérence faible/forte filtrée par un seuil de confiance fixe (`tau`). |
| FlexMatch | [scripts/flexmatch.py](scripts/flexmatch.py) | Zhang et al., 2021 | FixMatch + seuillage **adaptatif par classe** (Curriculum Pseudo Labeling), pour corriger le biais envers les classes "faciles". |
| MixMatch | [scripts/mixmatch.py](scripts/mixmatch.py) | Berthelot et al., 2019 | Pas d'augmentation forte : guessing par moyenne de K augmentations faibles + sharpening + Mixup entre labellisé et non labellisé. |
| EfficientMatch | [scripts/efficientmatch.py](scripts/efficientmatch.py) | (contribution de ce dépôt) | FixMatch + canal de **Mixup filtré** par le masque de confiance dur, entre le batch labellisé et le batch non labellisé faiblement augmenté. |

D'autres algorithmes pourront être ajoutés plus tard, chacun comme un nouveau script autonome du
même type.

## Structure du dépôt

```
scripts/
    fixmatch.py      # script autonome : python fixmatch.py [options]
    flexmatch.py     # script autonome : python flexmatch.py [options]
    mixmatch.py      # script autonome : python mixmatch.py [options]
    efficientmatch.py  # script autonome : python efficientmatch.py [options]
    analyze.py       # calcule AUC, itérations jusqu'à seuil (métriques absentes des logs bruts)
    models.py        # WideResNet (partagé)
    ema.py           # EMA des poids (partagée)
    evaluate.py      # évaluation top-1 (partagée)
    data.py          # chargement CIFAR-10 + split + wrapping de dataset (partagé, aucune
                      # augmentation ni composition de vues -- cf. chaque script)
    requirements.txt
    README.md        # détails d'utilisation, options CLI complètes
```

## Utilisation

```powershell
pip install -r scripts/requirements.txt

python scripts/fixmatch.py
python scripts/flexmatch.py --n-labels 250
python scripts/mixmatch.py --alpha-mix 0.5 --K 65536
python scripts/efficientmatch.py --lambda-mix 0.5
```

Tous les hyperparamètres sont exposés en flags CLI, propres à chaque script (`--help` pour la liste
complète). Voir [scripts/README.md](scripts/README.md) pour le détail des options et des exemples.

## Métriques suivies

Chaque run journalise, à intervalles réguliers (`--eval-every`), l'accuracy top-1 sur le jeu de test
(poids EMA) et la perte, dans `./logs/<algo>_cifar10_n<n_labels>_K<K>_seed<seed>[_<tag>].json` -- un
format commun à tous les scripts.

Les métriques du papier proprement dites (AUC normalisée sur `[0, K]`, itérations pour atteindre un
seuil de performance, avec report de la dernière valeur pour les runs incomplets) ne sont PAS
calculées pendant l'entraînement : elles sont dérivées après coup des logs bruts par
[scripts/analyze.py](scripts/analyze.py).

## Licence

[MIT](LICENSE)
