# Runs interrompus / compliqués à relancer plus tard

Suivi des runs arrêtés manuellement parce qu'ils divergent, stagnent ou sont anormalement lents. Chaque entrée note l'état au moment de l'arrêt pour pouvoir reprendre l'analyse plus tard.

## flexmatch — SVHN, 250 labels, seed 2312, target_acc 0.90

- **Commande** : `python flexmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.90`
- **Arrêté le** : 2026-09-13
- **État à l'arrêt** : 208 évaluations, ~104 000 steps, **97.03 minutes** écoulées
- **Accuracy** : dernière valeur 82.60%, max atteint 83.54% — n'a jamais approché le seuil de 90%
- **Comparaison** : sur la même seed, efficientmatch_2 a atteint 90.04% en 6.76 min et fixmatch 90.01% en 7.82 min — flexmatch était donc ~12x plus lent sans même atteindre l'objectif
- **Diagnostic** : progression très lente et en ralentissement continu (+0.32 point sur les dernières 37 minutes avant l'arrêt) — semble se diriger vers un plateau autour de 83-84%, pas de divergence franche mais convergence anormalement lente comparée aux autres méthodes
- **À refaire** : oui, si besoin de comparer flexmatch sur cette config — envisager de le laisser tourner plus longtemps sans `target_acc` ou d'enquêter sur la cause de la lenteur (seuillage adaptatif ? learning rate ?)

## flexmatch — SVHN, 250 labels, seed 0308, target_acc 0.90

- **Commande** : `python flexmatch.py --dataset svhn --num_labeled 250 --seed 0308 --target_acc 0.90`
- **Arrêté le** : 2026-09-13
- **État à l'arrêt** : 60 évaluations, ~29 500 steps, **28.21 minutes** écoulées
- **Accuracy** : dernière valeur 82.12%, max atteint 82.38% — n'a jamais approché le seuil de 90%
- **Comparaison** : sur la même seed, efficientmatch_2 a atteint 90.24% en 4.55 min (step 8500) et fixmatch 90.07% en step 6500 — flexmatch était donc très en retard, plafonnant nettement plus bas
- **Diagnostic** : même pattern que sur seed 2312 (voir entrée ci-dessus) — progression lente, plafond autour de 82-83%, aucune divergence brutale mais convergence anormalement lente et bloquée bien en dessous des autres méthodes. Confirme que le problème de flexmatch sur cette config (SVHN 250 labels, target 90%) est reproductible sur plusieurs seeds, pas un accident isolé
- **À refaire** : oui — le comportement récurrent sur 2 seeds suggère un vrai problème méthodologique (hyperparamètres flexmatch mal adaptés à SVHN/250 labels ?) plutôt qu'une variance aléatoire, à investiguer avant de relancer

## flexmatch — SVHN, 250 labels, seed 2701, target_acc 0.90

- **Commande** : `python flexmatch.py --dataset svhn --num_labeled 250 --seed 2701 --target_acc 0.90`
- **Arrêté le** : 2026-09-13 (règle fixée par l'utilisateur : kill après 30 min de run)
- **État à l'arrêt** : 81 évaluations, ~40 000 steps, **37.99 minutes** écoulées (dépassement du timeout de 30 min — le watcher automatique a eu un délai de rattrapage sur le gros fichier de log et n'a pas tué à temps, arrêt fait manuellement)
- **Accuracy** : dernière valeur 85.06%, max atteint 85.10% — n'a jamais atteint le seuil de 90%, mais nettement mieux que sur les 2 autres seeds (82-84%)
- **Comparaison** : sur la même seed, efficientmatch_2 a atteint 90.02% en 10000 steps et fixmatch 90.18% en 8000 steps
- **Diagnostic** : 3e seed sur 3 où flexmatch n'atteint pas 90% sur cette config (SVHN 250 labels) — confirme le problème méthodologique déjà noté, même si cette seed progressait mieux que les 2 précédentes (85% vs 82-84% de plafond)
- **À refaire** : oui, avec le même besoin d'investigation que les 2 entrées précédentes
