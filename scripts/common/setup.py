"""Model, EMA, optimiser and learning-rate schedule.

Identical for every method: the comparison is between training objectives, so everything around
them is held fixed. SGD with Nesterov momentum 0.9 at learning rate 0.03, decayed by the rescaled
FixMatch cosine over 2^20 iterations whatever the run's own length.
"""
import logging

import torch

from models import build_model
from utils import build_lr_scheduler
from ema import EMA

logger = logging.getLogger(__name__)


def select_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def enable_runtime_optimizations():
    """Runtime settings that change speed, never the algorithm (see docs/flops.md): cuDNN picks the
    fastest convolution algorithm for our fixed shapes, and float32 matrix products run in TF32."""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def build_model_and_optimizer(args, num_classes, device, compile_model=True):
    """Returns (model, base_model, eval_model, ema, optimizer, scheduler).

    `model` is what the training loop calls, possibly wrapped by torch.compile. `base_model` is the
    uncompiled module: compiling prefixes the state-dict keys ("_orig_mod.conv1.weight") and the EMA
    looks keys up by name, so the EMA has to be updated from `base_model`. Both share the same
    parameters. `eval_model` is the separate copy the EMA weights are written into for evaluation,
    or `model` itself when --use_ema is off, in which case `ema` is None.

    Call this after the labeled/unlabeled split: the weight initialisation draws from the RNG at
    the point it did before this code was factored out (see common/data.py).
    """
    model = build_model(args.model, num_classes, args.widen_factor, args.depth).to(device)
    if args.optimized:
        model = model.to(memory_format=torch.channels_last)

    if args.use_ema:
        eval_model = build_model(args.model, num_classes, args.widen_factor, args.depth).to(device)
        if args.optimized:
            eval_model = eval_model.to(memory_format=torch.channels_last)
        ema = EMA(model, args.ema_decay)
    else:
        eval_model, ema = model, None

    base_model = model
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = build_lr_scheduler(optimizer, args.total_steps, schedule=args.lr_schedule)

    # Compilation is worth its one-off cost only for fixed tensor shapes; Fast FixMatch, whose batch
    # size changes every iteration, is the counter-example measured in docs/fast_fixmatch.md.
    # RegMixMatch passes compile_model=False unless --static_shapes makes its own shapes fixed.
    if compile_model and args.optimized and torch.cuda.is_available():
        try:
            import triton  # noqa: F401
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile active (mode=reduce-overhead)")
        except ImportError:
            pass

    return model, base_model, eval_model, ema, optimizer, scheduler
