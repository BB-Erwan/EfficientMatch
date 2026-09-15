# Runs interrompus / compliqués à relancer plus tard

Suivi des runs arrêtés manuellement parce qu'ils divergent, stagnent ou sont anormalement lents. Chaque entrée note l'état au moment de l'arrêt pour pouvoir reprendre l'analyse plus tard.

## flexmatch — CIFAR-100, 10000 labels, seed 2312, target_acc 0.60, test_period 256

- **Commande** : `python flexmatch.py --dataset cifar100 --widen_factor 8 --num_labeled 10000 --seed 2312 --target_acc 0.60 --test_period 256`
- **Arrêté le** : 2026-09-15 (règle fixée par l'utilisateur : kill après 2h de run, watchdog automatique)
- **État à l'arrêt** : 38 évaluations, step 9472, **135.20 minutes** écoulées (dépassement du seuil de 2h)
- **Accuracy** : dernière valeur et max identiques, 56.28% — n'a jamais atteint le seuil de 60%
- **Comparaison** : sur la même seed/config, fixmatch a atteint 60.08% en 109 minutes (step 11520) — flexmatch était donc déjà en retard et n'a pas rattrapé avant le timeout
- **Diagnostic** : pas de divergence franche (pas de chute d'accuracy ni de NaN), simplement une convergence plus lente que fixmatch sur ce budget de labels — cohérent avec le pattern déjà observé pour flexmatch dans cette session (toujours parmi les méthodes les plus lentes, cf. les autres entrées de ce document)
- **Incident technique associé** : le watchdog automatique (script de sondage sur le fichier de log) a mis du temps à détecter le dépassement et sa commande `Stop-Process` n'a tué qu'un des deux processus (le wrapper `conda run` et le processus `python` réel) — arrêt complété manuellement. Un `TIMEOUT` précédent sur fixmatch/seed 2312 (juste avant ce run) s'est aussi déclenché en faux positif après que ce run avait déjà légitimement terminé, à cause d'un délai de détection — aucune conséquence dans ce cas précis, mais à garder à l'esprit si d'autres `TIMEOUT` apparaissent dans les logs de cette session
- **À refaire** : oui, si besoin de comparer flexmatch sur CIFAR-100/10000 labels — envisager de le laisser tourner plus longtemps sans `target_acc` pour voir où il plafonne réellement

## fixmatch — CIFAR-100, 10000 labels, seed 0308, target_acc 0.60, test_period 256

- **Commande** : `python fixmatch.py --dataset cifar100 --widen_factor 8 --num_labeled 10000 --seed 0308 --target_acc 0.60 --test_period 256`
- **Arrêté le** : 2026-09-15 (règle utilisateur : kill après 2h, watchdog automatique)
- **État à l'arrêt** : 37 évaluations, step 9216, **117.27 minutes** écoulées (dépassement du seuil de 2h)
- **Accuracy** : dernière valeur et max identiques, 58.37% — n'a jamais atteint le seuil de 60%, mais en était très proche
- **Comparaison** : sur seed 2312, fixmatch avait atteint 60.08% en 109 min (step 11520) — cette seed (0308) est donc légèrement plus lente à converger, sans être franchement anormale (juste au-dessus du timeout de 2h)
- **Diagnostic** : pas de divergence, juste une variance normale de seed à seed qui l'a fait passer de justesse au-dessus du seuil de 2h — contrairement à l'entrée flexmatch précédente, ce n'est probablement pas un vrai problème méthodologique, plutôt de la malchance sur le timing du timeout
- **Incident technique associé** : même bug de kill multi-PID que pour flexmatch/seed 2312 ci-dessus (le `Stop-Process` du watchdog n'a pas fonctionné, arrêt complété manuellement)
- **À refaire** : oui si besoin d'une comparaison complète — le run était probablement à quelques minutes d'atteindre 60% naturellement, un timeout légèrement plus long (ex. 2h15) suffirait sans doute

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

## flexmatch — SVHN, 250 labels, seed 2312, target_acc 0.90 (retenté après fix thresh_warmup)

- **Commande** : `python flexmatch.py --dataset svhn --num_labeled 250 --seed 2312 --target_acc 0.90`
- **Contexte** : relancé le 2026-09-15 après correction d'un bug réel dans `thresh_warmup` (voir `EXPERIMENT_LOG.md`/historique de session) — `torch.bincount(selected_label[selected_label != -1] + 1, ...)` filtrait les entrées `-1` avant comptage, rendant `--thresh_warmup True/False` strictement équivalents. Corrigé pour laisser le bin 0 (`-1`) dominer en début d'entraînement, ce qui doit rendre le seuil permissif puis progressivement plus strict, comme dans l'implémentation de référence
- **Arrêté le** : 2026-09-15 (décision manuelle, plateau confirmé sans limite de temps fixée à l'avance)
- **État à l'arrêt** : 228 évaluations, step 113 500, **106.97 minutes** écoulées
- **Accuracy** : dernière valeur 83.22%, max atteint 83.76% (step 110500) — plafond stable depuis le step ~22000 (environ 85 minutes de plateau quasi plat), n'a jamais approché le seuil de 90%
- **Comparaison avec les autres méthodes (même seed)** : regmixmatch 90.51% (10.3 min), mixmatch 90.27% (6.8 min), efficientmatch 90.05% (28.5 min), efficientmatch_2 90.04% (6.8 min), fixmatch 90.01% (7.8 min) — toutes atteignent 90% largement avant que flexmatch ne plafonne
- **Diagnostic** : le fix a clairement amélioré la vitesse de convergence initiale par rapport aux tentatives précédentes (montée à 80%+ dès le step ~15000-22000 au lieu d'un démarrage très lent), et le mask_ratio est passé de ~0.2 à ~0.83-0.87 une fois le warmup effectif — mais le plafond final (~83-84%) reste quasiment identique aux 3 tentatives précédentes (82.60%, 82.12%, 85.06% sur les 3 seeds avant le fix). Ceci confirme que le plafond de flexmatch sur cette config (SVHN, 250 labels, target 90%) est un vrai problème méthodologique indépendant du bug de warmup — pas résolu par ce correctif
- **Incident technique associé** : deux faux positifs de surveillance pendant ce run — (1) le monitor basé sur `ps -p <pid>` a signalé à tort "PROCESS EXITED" alors que le process tournait toujours (mismatch PID Git Bash/MSYS vs PID Windows réel pour un process externe lancé via `conda run`), corrigé en basculant la détection sur la présence du PID dans `nvidia-smi --query-compute-apps`; (2) le stdout du script est resté totalement vide dans le fichier de log tout du long (0 octet même après arrêt du process) à cause du buffering bloc de Python redirigé vers fichier — la progression a dû être suivie en lisant directement le JSON de métriques plutôt que le log
- **À refaire** : oui si on veut confirmer le plafond sur les 2 autres seeds (0308, 2701) avec le fix appliqué — mais l'hypothèse la plus probable reste un problème d'hyperparamètres/seuillage adaptatif spécifique à SVHN/250 labels plutôt qu'un bug de code restant

## mixmatch — CIFAR-10, 250 labels, seed 2312, target_acc 0.80

- **Commande** : `python mixmatch.py --dataset cifar10 --num_labeled 250 --seed 2312 --target_acc 0.80`
- **Statut** : run historique (pas relancé dans cette session), résultat déjà présent dans `results/cifar10-labeled-250-seed-2312/mixmatch_ema_metrics.json`
- **Résultat** : 1658 évaluations, 828 500 steps, **779.67 minutes (~13h)**, plafonné à 78.42% max — n'a jamais atteint 80%
- **Problème identifié et confirmé chiffré** : ce run a été exécuté sur un **GPU différent** de celui utilisé pour toutes les autres mesures de temps de cette session (RTX 5060 Ti) — vraisemblablement l'A4000 documenté dans `BENCHMARK_RESULTS.md`/`PROJECT_SPEC.md`. Vérification : le temps par itération de mixmatch sur cette seed est de **56.46 ms/step**, contre **18.80 ms/step** (seed 0308) et **18.81 ms/step** (seed 2701) — ces deux dernières quasi identiques entre elles (écart de 0.05%), confirmant qu'elles ont tourné sur le même GPU cette session, tandis que la seed 2312 est **~3x plus lente par itération**. Son temps d'exécution **n'est donc pas comparable** aux temps des autres méthodes/seeds mesurés cette session, malgré son utilisation dans plusieurs comparaisons de cette conversation
- **À refaire** : oui — relancer sur le même GPU (RTX 5060 Ti) que le reste des expériences de cette session pour obtenir un temps comparable. Les comparaisons faites jusqu'ici impliquant ce run spécifique (ex. "mixmatch a mis 13h sans converger") restent qualitativement informatives (il n'a de toute façon jamais atteint le seuil) mais le chiffre de temps exact est à prendre avec précaution
