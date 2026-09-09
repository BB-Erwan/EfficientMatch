"""CLI d'entraînement pour FixMatch.

Exemples :
    python train.py --algo fixmatch
    python train.py --algo fixmatch --n-labels 250 --K 65536 --tau 0.9
    python train.py --algo fixmatch --set weight_decay=1e-3
"""
import argparse
import ast
import sys

from algorithms import ALGORITHMS
from config import BASE_CONFIG, DATASET_DEFAULTS, build_config, compute_log_path
from engine import run_experiment


def _add_argument(parser, key, value):
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        parser.add_argument(flag, dest=key, action=argparse.BooleanOptionalAction, default=value)
    elif key == "debug_subset_size":
        parser.add_argument(
            flag, dest=key, type=str, default="none" if value is None else str(value),
            help="entier, ou 'none' pour désactiver (défaut: none)",
        )
    else:
        parser.add_argument(flag, dest=key, type=type(value), default=value)


def build_arg_parser(defaults):
    parser = argparse.ArgumentParser(
        description="Entraînement SSL en ligne de commande.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo", required=True, choices=sorted(ALGORITHMS), help="algorithme à entraîner")
    for key, value in sorted(defaults.items()):
        if key in ("algo", "log_path"):
            # "algo" a déjà un flag dédié ci-dessus (conflit argparse sinon) ; "log_path" est
            # toujours recalculé après coup à partir de (algo, dataset, n_labels, K, seed, tag),
            # cf. parse_args -- l'exposer en flag CLI serait trompeur (la valeur passée serait ignorée).
            continue
        _add_argument(parser, key, value)
    parser.add_argument(
        "--set", dest="extra_overrides", action="append", default=[], metavar="CLE=VALEUR",
        help="surcharge libre d'un paramètre non exposé explicitement (répétable), ex: --set weight_decay=1e-3",
    )
    return parser


def _parse_extra_overrides(pairs):
    overrides = {}
    for pair in pairs:
        key, _, raw_value = pair.partition("=")
        try:
            overrides[key] = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            overrides[key] = raw_value
    return overrides


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--algo", choices=sorted(ALGORITHMS))
    pre_parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS))
    known, _ = pre_parser.parse_known_args(argv)
    if known.algo is None:
        build_arg_parser(BASE_CONFIG).parse_args(argv)  # --algo manquant -> affiche l'aide et sort
        raise SystemExit(2)

    # Le dataset est pré-lu ici pour que les VALEURS PAR DÉFAUT de --num-classes/--weight-decay
    # affichées dans --help et utilisées si l'utilisateur ne les passe pas explicitement reflètent
    # déjà le dataset demandé (cf. config.DATASET_DEFAULTS) -- sinon `--dataset cifar100` sans
    # `--weight-decay` explicite se retrouverait silencieusement avec le weight_decay de CIFAR-10.
    pre_overrides = {"dataset": known.dataset} if known.dataset else {}
    defaults = build_config(known.algo, pre_overrides)
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    cfg = dict(defaults)
    for key in defaults:
        if key == "log_path":
            continue  # pas de flag CLI dédié (cf. build_arg_parser) -- recalculé plus bas
        cfg[key] = getattr(args, key)
    if isinstance(cfg["debug_subset_size"], str):
        cfg["debug_subset_size"] = None if cfg["debug_subset_size"].lower() == "none" else int(cfg["debug_subset_size"])
    cfg.update(_parse_extra_overrides(args.extra_overrides))
    cfg["algo"] = args.algo
    # Recalculé en dernier : dépend de (algo, dataset, n_labels, K, seed), tous potentiellement
    # modifiés par les flags CLI ou par --set ci-dessus.
    cfg["log_path"] = compute_log_path(cfg)
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    algo_module = ALGORITHMS[cfg["algo"]]
    run_experiment(cfg, algo_module)


if __name__ == "__main__":
    main()
