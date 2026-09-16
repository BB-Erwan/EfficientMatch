# Handoff — reprendre la session ici

Ce document résume tout ce qu'une nouvelle session Claude Code doit savoir pour reprendre
exactement où cette session s'est arrêtée : conventions établies, outils construits, pièges
rencontrés, et travail en attente. À lire avant de relancer quoi que ce soit sur ce repo.

**État au moment de la rédaction** : GPU libre, aucun run en cours, aucun process résiduel.

---

## 1. Contexte du projet

Repo de recherche en semi-supervised learning (`EfficientMatch`). Méthodes comparées : fixmatch,
flexmatch, mixmatch, sequencematch, efficientmatch / efficientmatch_2 / efficientmatch_3
(variantes maison), regmixmatch (porté depuis `hhrd9/regmixmatch`). Architecture : WideResNet-28-x
(x = widen_factor), backbone `build_model()` dans `models.py`.

**Périmètre retenu pour l'article** (imposé par l'utilisateur) : seeds **2312, 0308, 2701**
uniquement, sur les configs **SVHN 250 labels**, **CIFAR-10 250 labels**, **CIFAR-10 4000 labels**,
**CIFAR-100 10000 labels** (+ CIFAR-100 2500 labels ajouté en cours de session pour des tests
ponctuels). Toute autre config présente dans `results/` est hors périmètre.

GPU de la machine : **RTX 5060 Ti, 8GB de VRAM seulement** — c'est la contrainte matérielle la
plus importante à connaître (voir §4).

---

## 2. Outil principal : `scripts/run_analysis.py`

**Construit spécifiquement pour ne plus jamais recoder un script d'analyse ponctuel.** Toujours
l'utiliser pour vérifier l'avancée d'un run ou comparer des méthodes — ne pas écrire de script
Python à la volée pour lire un fichier de métriques.

```
python scripts/run_analysis.py compare --dataset cifar10 --num_labeled 250 --seed 2312
python scripts/run_analysis.py at-acc --dataset cifar100 --num_labeled 10000 --seed 2312 --like efficientmatch_3_ema_wf4
python scripts/run_analysis.py status --dataset cifar10 --num_labeled 250 --seed 2312 --method efficientmatch_3_ema --tail 8
```

- `compare` : tableau complet trié par accuracy, avec temps brut/corrigé (overhead d'éval retiré) et FLOPs totaux.
- `at-acc` : compare toutes les méthodes **au même niveau d'accuracy** qu'une méthode de référence
  (`--like`) ou qu'un seuil (`--target`) — **c'est la commande à utiliser par défaut** quand
  l'utilisateur demande "où on en est" / "des nouvelles" sur un run en cours, pas `status` seul.
- `status --tail N` : affiche les N dernières évaluations, utile pour juger si un run sans
  `target_acc` a plafonné.
- Détecte l'architecture **par fichier** (suffixe `_wfN` dans le nom), pas par dossier — un même
  dossier CIFAR-100 peut légitimement mélanger des résultats WRN-28-4 et WRN-28-8.
- Si l'outil manque une capacité (nouvelle architecture, nouvelle métrique), **l'étendre**, ne pas
  contourner avec un script jetable.

Autres outils construits cette session :
- `scripts/measure_eval_time.py` : mesure isolée du coût d'une évaluation (avec copie EMA incluse).
- `scripts/ghost_method.py` : mesure **in situ** (même harnais qu'un vrai run — dataloaders, EMA,
  torch.compile — mais la boucle d'entraînement ne fait rien) pour valider les mesures isolées.
- `scripts/flops_analysis.py` : FLOPs/itération par méthode et architecture (`--widen-factor`).

---

## 3. Conventions établies (à respecter sans qu'on ait besoin de le redemander)

- **Sweep multi-seeds : toujours seed-first** (toutes les méthodes sur une seed avant de passer à
  la seed suivante), pas method-first.
- **Ne jamais annoncer le démarrage d'un run** dans un sweep séquentiel — seulement les fins (succès,
  crash, divergence). Silence sur les points de progression intermédiaires sauf si l'utilisateur
  scrolle explicitement pour un "où on en est".
- **À chaque fin de run, comparer automatiquement** avec les autres méthodes de la même config
  (tableau markdown), sans attendre qu'on le demande.
- **Pour un run en cours, comparer "à accuracy équivalente"** (temps/step où les autres méthodes
  ont atteint le même niveau d'accuracy), jamais le temps brut du run en cours contre le temps
  final des autres à une accuracy différente.
- **Ne jamais relancer une expérience qui a déjà un résultat exploitable** sauf demande explicite —
  vérifier l'inventaire avant de lancer un sweep.
- **Avant de lancer un nouveau run, vérifier que le GPU est libre** (`nvidia-smi
  --query-compute-apps=pid,process_name --format=csv,noheader`).
- **Entre deux runs d'une séquence, faire une vérification manuelle rigoureuse** des process
  restants (voir §4, `TaskStop` n'est pas fiable) — ne pas se contenter de l'automatisation.
- Toute comparaison chiffrée entre méthodes = **tableau markdown**, jamais une liste à puces ou un
  paragraphe.

---

## 4. Pièges rencontrés cette session (ne pas les redécouvrir)

### `TaskStop` ne tue pas toujours l'arborescence de process réelle
Sur Windows/Git Bash, arrêter une tâche qui lance `bash script.sh` (qui lui-même fait
`conda run -n ENV python train.py`) via `TaskStop` a échoué **deux fois** à tuer le process réel :
le `bash.exe` orchestrateur et/ou le `conda.exe`/`python.exe` enfant continuaient de tourner en
arrière-plan, invisibles pour `nvidia-smi` tant qu'ils n'avaient pas (re)pris la main sur le GPU.
**Toujours vérifier manuellement après un `TaskStop`** :
```powershell
Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -match 'fixmatch\.py|flexmatch\.py|mixmatch\.py|regmixmatch\.py|sequencematch\.py|efficientmatch(_2|_3)?\.py'
} | Select-Object ProcessId, Name, CommandLine | Format-Table -AutoSize -Wrap
```
et tuer explicitement (`Stop-Process -Id <pid> -Force`) tout ce qui apparaît, même si `nvidia-smi`
dit "GPU clean". Mémoire dédiée : `taskstop-orphaned-processes.md`.

### WRN-28-8 sature la VRAM (8GB) sur CIFAR-100 avec certaines méthodes
`regmixmatch` sur CIFAR-100/WRN-28-8 avec `torch.compile(reduce-overhead)` a tourné à ~20s/step
(au lieu de ~0.6s/step attendu) — VRAM à 96% (7838/8151 MiB), GPU à 100% d'utilisation mais
seulement 37W de conso (signature de memory-thrashing, pas de calcul réel). **WideResNet-28-4 est
devenu l'architecture de facto pour CIFAR-100** sur cette machine : fixmatch, flexmatch, mixmatch,
regmixmatch et efficientmatch_3 ont tous été retestés en `--widen_factor 4` avec succès (facteur de
gain de temps ~2.5x à ~6x selon la méthode par rapport à WF8, à accuracy équivalente — voir
`FLOPS_RESULTS.md`). Détail incident dans `RUNS_TO_REVISIT.md`.

### Convention de nommage des fichiers de résultats par architecture
Tous les scripts (`fixmatch.py`, `flexmatch.py`, `mixmatch.py`, `regmixmatch.py`,
`efficientmatch.py`/`_2`/`_3`, `sequencematch.py`) ajoutent maintenant un suffixe `_wf{N}` au nom
du fichier de résultats quand `--widen_factor` diffère de la valeur par défaut du script (2). **Le
suffixe doit être en toute fin de nom de fichier** pour que `run_analysis.py` (regex
`_wf(\d+)$`) le détecte — un fichier renommé manuellement avec du texte après le suffixe (ex.
`..._wf4_target60_metrics.json`) casse la détection ; mettre le suffixe wf en dernier
(`..._target60_wf4_metrics.json`). Un fichier CIFAR-100 sans suffixe est un run historique WRN-28-8
(avant l'ajout de cette convention) — `run_analysis.py --default_widen_factor` permet de l'ajuster.

**Attention à la collision de noms** : avant de lancer un run avec un `--widen_factor` ou un
paramètre non-standard sur une config déjà utilisée, vérifier que le script gère bien ce paramètre
dans son nom de fichier de sortie — sinon il écrase silencieusement un résultat existant. Toujours
vérifier `ls` du dossier de résultats avant un lancement inhabituel.

### Autres pièges déjà documentés dans les mémoires persistantes
- `pgrep -f` peu fiable sur l'arborescence de process Windows/Git Bash.
- `tail -f | grep -m1` ne se termine pas de façon fiable (préférer les boucles de polling).
- Le stdout des scripts Python redirigé vers un fichier est bufferisé par blocs — **lire le JSON de
  métriques directement plutôt que le log texte** pour connaître la progression réelle.
- `conda run ... python -c "<code multi-lignes>"` échoue (conda ne supporte pas les retours à la
  ligne dans les arguments) — écrire le code dans un fichier temporaire et l'exécuter.
- `nvidia-smi --query-compute-apps` peut donner un faux "aucun process" de façon transitoire même
  quand un run tourne activement (observé plusieurs fois) — ne pas conclure à un crash sur un seul
  relevé ; vérifier avec `tasklist`/`Get-CimInstance` en complément.

---

## 5. Documents de référence à connaître

- **`RUNS_TO_REVISIT.md`** — tous les runs interrompus/compliqués avec diagnostic détaillé
  (bug thresh_warmup, runs tués au timeout, incident VRAM regmixmatch/CIFAR-100, etc.).
- **`FLOPS_RESULTS.md`** — FLOPs/itération par méthode et architecture (WRN-28-2/4/8), coût d'une
  évaluation par architecture, analyse de l'overhead d'évaluation sur le temps mesuré.
- **`SEEDS_2312_308_2701_INVENTORY.md`** — inventaire des résultats sur les 3 seeds retenues.
  ⚠️ **Ce document n'a pas été remis à jour depuis le pivot vers WRN-28-4 sur CIFAR-100** (section
  CIFAR-100 10000 labels toujours en WRN-28-8, avec les anciens échecs fixmatch/flexmatch/seed 308
  qui ont depuis été corrigés en WF4). À rafraîchir avant toute utilisation pour l'article.
- **`EXPERIMENT_LOG.md`**, **`CIFAR10_4000LABELS_RESULTS.md`** — narratif et résultats détaillés
  d'expériences antérieures (CIFAR-10 4000 labels).

---

## 6. Inventaire de couverture efficientmatch_3 (dernier état vérifié)

| Config | Seed 2312 | Seed 308 | Seed 2701 |
|---|---|---|---|
| SVHN 250 labels | ✅ mu=3, mu=5 | ✅ mu=3, mu=5 | ✅ mu=3, mu=5 |
| CIFAR-10 250 labels | ✅ mu=1,3,5,7 | ✅ mu=3, mu=5 | ✅ mu=3, mu=5 |
| CIFAR-10 4000 labels | ❌ jamais testé | ❌ jamais testé | ❌ jamais testé |
| CIFAR-100 10000 labels | ✅ mu=3, mu=5 (WF4) + ancien WF8 | ❌ jamais testé | ❌ jamais testé |
| CIFAR-100 2500 labels | ✅ mu=3 (WF4) | ❌ jamais testé | ❌ jamais testé |

## 7. Couverture fixmatch/flexmatch/mixmatch/regmixmatch sur CIFAR-100 10000 labels (WF4)

| Méthode | Seed 2312 | Seed 308 | Seed 2701 |
|---|---|---|---|
| fixmatch | ✅ 60.00% | ✅ 60.01% | ❌ jamais lancé en WF4 |
| flexmatch | ✅ 60.29% | ✅ 60.04% | ❌ jamais lancé en WF4 |
| mixmatch | ✅ 60.01% (target60) + 64.26% (plafond sans target) | ✅ 60.38% (WF8, déjà réussi, non relancé en WF4) | ✅ 60.18% (WF8, déjà réussi, non relancé en WF4) |
| regmixmatch | ✅ 60.07% | ❌ jamais lancé | ❌ jamais lancé |

---

## 8. Travail en attente, par priorité

L'utilisateur a explicitement mis en pause le plan de "combler tous les trous efficientmatch_3"
(§6) pour d'abord tester efficientmatch_3 sur CIFAR-100 10000/seed 2312 (fait, voir §6-7) — **ce
plan n'a pas été formellement relancé après ça**. Ne pas reprendre automatiquement sans confirmer
avec l'utilisateur quelle priorité il souhaite.

**Priorité annoncée avant la pause** (sweep CIFAR-100 10000 labels, target 60%, test_period 256,
WF4, seed-first) :
1. `regmixmatch` — seed 308 (jamais lancé)
2. `fixmatch` — seed 2701 (jamais lancé en WF4)
3. `flexmatch` — seed 2701 (jamais lancé en WF4)
4. `regmixmatch` — seed 2701 (jamais lancé)
(mixmatch déjà réussi sur 308 et 2701 en WF8, non relancé par choix — "ne pas refaire ce qui a déjà
un résultat")

**Ensuite, si confirmé par l'utilisateur** — combler les trous efficientmatch_3 (§6) :
- CIFAR-10 4000 labels, 3 seeds (mu=3 par défaut, potentiellement mu=5 aussi vu les bons résultats
  ailleurs)
- CIFAR-100 10000 labels, seeds 308 et 2701 (WF4)
- CIFAR-100 2500 labels, seeds 308 et 2701 (WF4)

**Maintenance documentaire en attente** :
- Rafraîchir `SEEDS_2312_308_2701_INVENTORY.md` avec tous les résultats WF4 (actuellement obsolète
  sur la partie CIFAR-100, voir §5).
- Une fois les trous efficientmatch_3 comblés, mettre à jour ce même document avec la couverture
  complète.

---

## 9. Style de réponse attendu par l'utilisateur

- Réponses en **français**, concises, orientées action.
- Tableaux markdown pour toute comparaison chiffrée.
- Ne pas halluciner de résultats non encore obtenus — attendre les notifications de fin de tâche
  réelles avant d'annoncer un chiffre.
- Le code (commentaires, docstrings) reste en anglais par convention du repo, mais les explications
  à l'utilisateur sont en français.
