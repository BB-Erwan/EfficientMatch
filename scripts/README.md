# Scripts CLI -- EfficientMatch

Chaque algorithme SSL est un script Python **autonome** : augmentations, split labellisé/non
labellisé, hyperparamètres (son propre `argparse`) et boucle d'entraînement vivent tous dans le même
fichier. FixMatch, FlexMatch, MixMatch et EfficientMatch sont implémentés pour l'instant. Rien n'est
partagé entre algorithmes sauf ce qui est strictement identique quel que soit l'algorithme
(architecture du modèle, EMA, évaluation, mécanique générique de chargement de données).

## Structure

```
scripts/
    fixmatch.py      # script autonome : python fixmatch.py [options]
    flexmatch.py     # script autonome : python flexmatch.py [options]
    mixmatch.py      # script autonome : python mixmatch.py [options]
    efficientmatch.py  # script autonome : python efficientmatch.py [options]
    analyze.py       # post-traitement des logs JSON -> AUC, itérations jusqu'à seuil, forward-fill
    models.py        # WideResNet-28-2 (architecture seule, partagée)
    ema.py           # EMA des poids (partagée)
    evaluate.py      # évaluation top-1 (partagée)
    data.py          # chargement CIFAR-10 + split + wrapping de dataset (mécanique générique,
                      # partagée -- AUCUNE augmentation ni composition de vues ici, cf. chaque script)
```

## Installation

```powershell
pip install -r scripts/requirements.txt
```

## Utilisation

```powershell
# FixMatch avec les hyperparamètres par défaut
python scripts/fixmatch.py

# Changer des hyperparamètres (voir --help pour la liste complète, propre à chaque algo)
python scripts/fixmatch.py --n-labels 250 --K 65536 --tau 0.9

# FlexMatch (seuil de confiance ADAPTATIF par classe -- Curriculum Pseudo Labeling)
python scripts/flexmatch.py --n-labels 250

# MixMatch (pas de seuil de confiance -- Mixup + moyenne/sharpening de K_aug vues faibles)
python scripts/mixmatch.py --alpha-mix 0.5

# EfficientMatch (FixMatch + canal de Mixup filtré par le masque de confiance dur)
python scripts/efficientmatch.py --lambda-mix 0.5

# --verbose affiche une ligne à chaque itération (loss, it/s) pour suivre la vitesse en direct,
# sans déclencher d'évaluation supplémentaire
python scripts/fixmatch.py --verbose

# Sous-ensemble de debug pour vérifier rapidement le pipeline
python scripts/fixmatch.py --debug-subset-size 2000 --K 200 --eval-every 50
```

Chaque run écrit sa configuration + ses logs (perte, accuracy) dans
`./logs/<algo>_cifar10_n<n_labels>_K<K>_seed<seed>[_<tag>].json` (le `tag` optionnel, via `--tag`,
sert à distinguer plusieurs runs qui partagent (algo, n_labels, K, seed)). Le fichier est réécrit à
chaque évaluation puis, à la fin du run, marqué `"status": "completed"` -- ce format est commun à
tous les scripts, c'est ce qui permet à `analyze.py` de comparer les algorithmes entre eux malgré des
boucles d'entraînement complètement indépendantes.

## Analyse des résultats (métriques du papier, absentes des logs bruts)

`analyze.py` calcule après coup : AUC normalisée sur `[0, K]`, report de la dernière valeur EMA
jusqu'à `K` pour les runs incomplets (pour que l'AUC reste comparable entre méthodes), et itérations
pour atteindre une fraction de l'accuracy asymptotique d'un algo de référence (FixMatch par défaut).

```powershell
python scripts/analyze.py --logs-dir ./logs --dataset cifar10 --n-labels 250 --K 131072 \
    --reference-algo fixmatch --threshold-frac 0.9 --out results.json
```
