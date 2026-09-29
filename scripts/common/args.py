"""Command-line options shared by every training script.

Only the options that are identical across all methods live here. Anything a method interprets its
own way -- the unlabeled ratio mu, whose default differs per method, the confidence threshold, the
Beta parameter of Mixup -- stays in that method's own script, where a reader looking for how the
method is configured will expect to find it.
"""
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def add_common_args(parser):
    """The 18 options every method accepts, with identical meaning and defaults."""
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    data.add_argument("--num_labeled", type=int, default=250,
                      help="Size of the labeled subset, split evenly across classes.")
    data.add_argument("--seed", type=int, default=42,
                      help="Seeds the labeled/unlabeled split and the weight initialisation. The paper "
                           "uses 2312, 0308 and 2701; the split it produces is what varies most between seeds.")

    model = parser.add_argument_group("model")
    model.add_argument("--model", type=str, default="wideresnet", choices=["wideresnet", "resnet18"],
                       help="Backbone architecture.")
    model.add_argument("--widen_factor", type=int, default=2,
                       help="WideResNet-28-{widen_factor}. The paper uses 2 for CIFAR-10 and SVHN, 4 for CIFAR-100.")
    model.add_argument("--depth", type=int, default=28,
                       help="WideResNet depth. Ignored for resnet18. Must satisfy (depth - 4) mod 6 == 0.")

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--weight_decay", type=float, default=5e-4,
                       help="SGD weight decay. Every run reported in the paper uses this default.")
    optim.add_argument("--max_steps", type=int, default=2**20,
                       help="Iterations actually run; the run is truncated here.")
    optim.add_argument("--total_steps", type=int, default=2**20,
                       help="Horizon the cosine learning-rate schedule decays over, independent of --max_steps. "
                            "Leaving it at 2^20 is what makes a short run follow the same schedule as a long one.")
    optim.add_argument("--lr_schedule", type=str, default="fixmatch_cosine",
                       choices=["fixmatch_cosine", "cosine_annealing"],
                       help="Rescaled FixMatch cosine (default) or torch's CosineAnnealingLR.")

    stop = parser.add_argument_group("stopping")
    stop.add_argument("--target_acc", type=float, default=None,
                      help="Stop as soon as the evaluated accuracy reaches this value.")
    stop.add_argument("--max_minutes", type=float, default=None,
                      help="Stop the run once it has been training for this many minutes: checked at each "
                           "evaluation, and counted from the first training step, so model compilation does "
                           "not eat into the budget. The paper caps every run at 120 minutes; "
                           "run_experiment.py passes it.")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--test_period", type=int, default=500,
                            help="Evaluate every N iterations. Every run in the paper uses 500.")
    evaluation.add_argument("--use_ema", type=str2bool, default=True,
                            help="Evaluate an exponential moving average of the weights rather than the "
                                 "training weights. All reported results use the EMA.")
    evaluation.add_argument("--ema_decay", type=float, default=0.999)
    evaluation.add_argument("--topk", type=int, default=1,
                            help="Also report top-k accuracy for every k up to this value.")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--optimized", type=str2bool, default=True,
                      help="Runtime optimisations that do not change the algorithm: channels-last memory "
                           "format, multi-worker loading, and torch.compile. See docs/flops.md.")
    misc.add_argument("--verbose", type=str2bool, default=False,
                      help="Print one line per iteration instead of only at evaluations.")
    misc.add_argument("--tag", type=str, default="",
                      help="Suffix appended to the result file name, to keep a special run from being "
                           "confused with the standard one (the paper's extended-horizon runs use 'unlimited').")
    return parser
