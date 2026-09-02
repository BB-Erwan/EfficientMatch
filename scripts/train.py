"""CLI d'entraînement pour les expériences SSL (FixMatch, FlexMatch, MixMatch, Fast FixMatch, EfficientMatch).

Exemples :
    python train.py --algo fixmatch
    python train.py --algo efficientmatch --n-labels 250 --K 65536 --no-use-amp
    python train.py --algo mixmatch --set weight_decay=1e-3 --set rampup_length=8000
"""
import argparse
import ast
import sys

from algorithms import ALGORITHMS
from config import BASE_CONFIG, build_config
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
    known, _ = pre_parser.parse_known_args(argv)
    if known.algo is None:
        build_arg_parser(BASE_CONFIG).parse_args(argv)  # --algo manquant -> affiche l'aide et sort
        raise SystemExit(2)

    defaults = build_config(known.algo)
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    cfg = dict(defaults)
    for key in defaults:
        cfg[key] = getattr(args, key)
    if isinstance(cfg["debug_subset_size"], str):
        cfg["debug_subset_size"] = None if cfg["debug_subset_size"].lower() == "none" else int(cfg["debug_subset_size"])
    cfg.update(_parse_extra_overrides(args.extra_overrides))
    cfg["algo"] = args.algo
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    algo_module = ALGORITHMS[cfg["algo"]]
    run_experiment(cfg, algo_module)


if __name__ == "__main__":
    main()
