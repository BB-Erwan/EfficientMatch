"""Mesure le nombre de FLOPs par itération (forward + backward) de chaque méthode SSL de
`scripts/` (fixmatch, flexmatch, mixmatch, efficientmatch, sequencematch), sur des tenseurs
factices (dummy tensors) de la bonne forme -- pas de CIFAR-10, pas de dataloader, pas de run
réel. Chaque fonction `<methode>_iter_flops` reproduit fidèlement la séquence d'appels au modèle
(nombre de forward, tailles de batch, no_grad éventuel) telle qu'elle apparaît dans
`scripts/<methode>.py`, afin que le résultat reflète le coût de calcul réel d'une itération et
pas une approximation.

Les FLOPs sont indépendants des données, de `--optimized`/`--amp`/`torch.compile` et du device :
seules les formes des tenseurs (batch, canaux, résolution) et l'architecture du modèle comptent.

Usage :
    python flops_analysis.py
    python flops_analysis.py --methods fixmatch efficientmatch
    python flops_analysis.py --out flops_metrics.json --md ../FLOPS_RESULTS.md
"""
import argparse
import json
import platform
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from models import WideResNet
from run_analysis import freematch_threshold_flops

NUM_CLASSES = 10
IMAGE_SIZE = 32
BATCH_SIZE_L = 64

# mu (ratio non labellisé / labellisé) et nombre de vues non labellisées par méthode, tels que
# codés en dur dans chaque script d'entraînement.
MU = {
    "fixmatch": 7,
    "flexmatch": 7,
    "mixmatch": 1,
    "efficientmatch": 3,
    "efficientmatch_2": 3,
    "efficientmatch_3": 3,
    "efficientmatch_freematch": 3,
    "efficientmatch_freematch_svhn": 3,
    "efficientmatch_freematch_c100": 3,
    "efficientmatch_3_mu1": 1,
    "efficientmatch_3_mu5": 5,
    "efficientmatch_3_mu7": 7,
    "efficientmatch_flex_mu2": 2,
    "sequencematch": 7,
    "regmixmatch": 7,
    "regmixmatch_mu3": 3,
}


def build_model(device, depth=28, widen_factor=2):
    model = WideResNet(depth=depth, widen_factor=widen_factor, num_classes=NUM_CLASSES).to(device)
    model.train()
    return model


def _dummy_images(n, device):
    return torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)


def _dummy_labels(n, device):
    return torch.randint(0, NUM_CLASSES, (n,), device=device)


def fixmatch_iter_flops(model, device):
    """1 forward no_grad (pseudo-labels, mu*B) + 1 forward+backward fusionné (B + mu*B)."""
    B, muB = BATCH_SIZE_L, MU["fixmatch"] * BATCH_SIZE_L
    x_l, y_l = _dummy_images(B, device), _dummy_labels(B, device)
    x_u_w, x_u_s = _dummy_images(muB, device), _dummy_images(muB, device)

    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            probs_u_w = F.softmax(model(x_u_w), dim=1)
        max_prob, pseudo = torch.max(probs_u_w, dim=1)
        mask = max_prob.ge(0.95).float()

        all_logits = model(torch.cat([x_l, x_u_s], dim=0))
        logits_l, logits_u_s = all_logits[:B], all_logits[B:]
        loss = F.cross_entropy(logits_l, y_l) + (
            mask * F.cross_entropy(logits_u_s, pseudo, reduction="none")
        ).mean()
        loss.backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops()


def flexmatch_iter_flops(model, device):
    """Même séquence d'appels au modèle que FixMatch (le seuillage par classe ne change ni le
    nombre de forwards, ni les tailles de batch)."""
    return fixmatch_iter_flops(model, device)


def mixmatch_iter_flops(model, device):
    """1 forward no_grad sur la paire de vues faibles (2*mu*B) + 1 forward+backward sur le
    mélange mixup (B + 2*mu*B)."""
    B, muB = BATCH_SIZE_L, MU["mixmatch"] * BATCH_SIZE_L
    x_l, y_l = _dummy_images(B, device), _dummy_labels(B, device)
    x_u_w_1, x_u_w_2 = _dummy_images(muB, device), _dummy_images(muB, device)
    beta_dist = torch.distributions.Beta(0.75, 0.75)

    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            x_u_pair = torch.cat([x_u_w_1, x_u_w_2], dim=0)
            all_logits = model(x_u_pair)
            probs_avg = (F.softmax(all_logits[:muB], dim=1) + F.softmax(all_logits[muB:], dim=1)) / 2
            probs_sharpened = probs_avg ** (1 / 0.5)
            probs_sharpened = probs_sharpened / probs_sharpened.sum(dim=1, keepdim=True)

        y_l_onehot = F.one_hot(y_l, NUM_CLASSES).float()
        probs_pair = torch.cat([probs_sharpened, probs_sharpened], dim=0)
        all_inputs = torch.cat([x_l, x_u_pair], dim=0)
        all_targets = torch.cat([y_l_onehot, probs_pair], dim=0)
        indices = torch.randperm(all_inputs.size(0), device=device)
        all_inputs, all_targets = all_inputs[indices], all_targets[indices]

        lam_x = torch.maximum(beta_dist.sample((B,)), 1 - beta_dist.sample((B,))).view(-1, 1, 1, 1).to(device)
        lam_u = torch.maximum(beta_dist.sample((2 * muB,)), 1 - beta_dist.sample((2 * muB,))).view(-1, 1, 1, 1).to(device)

        mixup_x = torch.lerp(all_inputs[:B], x_l, lam_x)
        mixup_u = torch.lerp(all_inputs[B:], x_u_pair, lam_u)
        all_mixup_inputs = torch.cat([mixup_x, mixup_u], dim=0)

        all_logits = model(all_mixup_inputs)
        logits_l, logits_u = all_logits[:B], all_logits[B:]
        loss = F.cross_entropy(logits_l, y_l) + F.mse_loss(
            F.softmax(logits_u, dim=1), all_targets[B:]
        )
        loss.backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops()


def efficientmatch_iter_flops(model, device, mu=None):
    """1 forward no_grad (pseudo-labels, mu*B) + 2 forward+backward fusionnés dans le même
    graphe (B + mu*B chacun) : un forward FixMatch-like et un forward sur le mélange mixup."""
    B, muB = BATCH_SIZE_L, (mu if mu is not None else MU["efficientmatch"]) * BATCH_SIZE_L
    x_l, y_l = _dummy_images(B, device), _dummy_labels(B, device)
    x_u_w, x_u_s = _dummy_images(muB, device), _dummy_images(muB, device)
    beta_dist = torch.distributions.Beta(0.75, 0.75)

    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            probs_u_w = F.softmax(model(x_u_w), dim=1)
        max_prob, pseudo = torch.max(probs_u_w, dim=1)
        mask = max_prob.ge(0.95).float()

        all_logits = model(torch.cat([x_l, x_u_s], dim=0))
        logits_l, logits_u_s = all_logits[:B], all_logits[B:]
        loss_supervised = F.cross_entropy(logits_l, y_l)
        loss_consistency = (mask * F.cross_entropy(logits_u_s, pseudo, reduction="none")).mean()

        all_inputs = torch.cat([x_l, x_u_w], dim=0)
        all_targets = torch.cat(
            [F.one_hot(y_l, NUM_CLASSES).float(), F.one_hot(pseudo, NUM_CLASSES).float()], dim=0
        )
        indices = torch.randperm(all_inputs.size(0), device=device)
        all_inputs, all_targets = all_inputs[indices], all_targets[indices]

        lam_x = torch.maximum(beta_dist.sample((B,)), 1 - beta_dist.sample((B,))).view(-1, 1, 1, 1).to(device)
        lam_u = torch.maximum(beta_dist.sample((muB,)), 1 - beta_dist.sample((muB,))).view(-1, 1, 1, 1).to(device)
        mixup_x = torch.lerp(all_inputs[:B], x_l, lam_x)
        mixup_u = torch.lerp(all_inputs[B:], x_u_w, lam_u)
        all_mixup_inputs = torch.cat([mixup_x, mixup_u], dim=0)

        all_logits = model(all_mixup_inputs)
        logits_l_mixup, logits_u_mixup = all_logits[:B], all_logits[B:]
        loss_mixup = (
            F.cross_entropy(logits_l_mixup, all_targets[:B].argmax(dim=1), reduction="none").mean()
            + F.cross_entropy(logits_u_mixup, all_targets[B:].argmax(dim=1), reduction="none").mean()
        )
        loss = loss_supervised + loss_consistency + loss_mixup
        loss.backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops()


def efficientmatch_2_iter_flops(model, device, mu=None):
    """Same forward-call sequence as efficientmatch: efficientmatch_2 only changes the mixup
    loss formula (a plain scalar-index cross_entropy vs. the two-term split above), which is an
    elementwise/reduction op that FlopCounterMode doesn't attribute matmul/conv FLOPs to -- the
    model() calls and their shapes are identical, so the FLOPs/iter are the same."""
    return efficientmatch_iter_flops(model, device, mu)


def efficientmatch_3_iter_flops(model, device, mu=None):
    """Same forward-call sequence as efficientmatch/efficientmatch_2: efficientmatch_3 only
    removes the .argmax(dim=1) from the mixup cross_entropy target (soft vs. hard labels), which
    doesn't change any model() call or tensor shape, so the FLOPs/iter are the same."""
    return efficientmatch_iter_flops(model, device, mu)


def efficientmatch_freematch_iter_flops(model, device, mu=None, num_classes=NUM_CLASSES, svhn_clamp=False):
    """efficientmatch_3 + FreeMatch's self-adaptive thresholding (--freematch_threshold).

    Same model() call sequence as efficientmatch_3 (measured by FlopCounterMode, which only counts
    conv/matmul FLOPs), PLUS the element-wise operations of the thresholding (EMA trackers, mean,
    max, product, compare), which FlopCounterMode does not see and are therefore counted by hand
    in run_analysis.freematch_threshold_flops (dependent on the number of classes and on the SVHN
    clamp)."""
    mu = mu if mu is not None else MU["efficientmatch_freematch"]
    model_flops = efficientmatch_iter_flops(model, device, mu)
    return model_flops + freematch_threshold_flops(mu * BATCH_SIZE_L, num_classes, svhn_clamp)


def regmixmatch_iter_flops(model, device, mu=None):
    """As actually run in this repo (--static_shapes True, --disab_cam True, the defaults used
    for every regmixmatch experiment): 2 forward+backward calls, no separate no_grad pseudo-label
    pass -- the first forward already yields the pseudo-labels as a slice of its own output.
    1) model(cat(x_l, x_u_w, x_u_s)): B + 2*mu*B.
    2) model(mixed_x) where mixed_x = resizemix(cat(x_l, x_u_s)): B + mu*B (static_shapes mixes
    over the full labeled+unlabeled population every step, not a confidence-filtered subset)."""
    B, muB = BATCH_SIZE_L, (mu if mu is not None else MU["regmixmatch"]) * BATCH_SIZE_L
    x_l, y_l = _dummy_images(B, device), _dummy_labels(B, device)
    x_u_w, x_u_s = _dummy_images(muB, device), _dummy_images(muB, device)

    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        all_logits = model(torch.cat([x_l, x_u_w, x_u_s], dim=0))
        logits_l, logits_w, logits_s = all_logits[:B], all_logits[B:B + muB], all_logits[B + muB:]

        sup_loss = F.cross_entropy(logits_l, y_l)
        with torch.no_grad():
            prob_w = F.softmax(logits_w.float(), dim=-1)
            max_probs, pseudo = torch.max(prob_w, dim=-1)
        ce_per_sample = F.cross_entropy(logits_s, pseudo, reduction="none")
        unsup_loss = ce_per_sample.mean()

        mixed_x = _dummy_images(B + muB, device)  # resizemix output: same shape as its conf_data_full input
        logits_mix = model(mixed_x)
        mixed_y = F.one_hot(_dummy_labels(B + muB, device), NUM_CLASSES).float()
        mix_loss = (F.log_softmax(logits_mix, dim=-1) * -mixed_y).sum(dim=-1).mean()

        loss = sup_loss + unsup_loss + mix_loss
        loss.backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops()


def sequencematch_iter_flops(model, device):
    """1 seul forward+backward sur la concaténation labeled + 3 vues non labellisées
    (faible/médium/forte) : B + 3*mu*B."""
    B, muB = BATCH_SIZE_L, MU["sequencematch"] * BATCH_SIZE_L
    x_l, y_l = _dummy_images(B, device), _dummy_labels(B, device)
    x_u_w, x_u_m, x_u_s = _dummy_images(muB, device), _dummy_images(muB, device), _dummy_images(muB, device)

    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        all_logits = model(torch.cat([x_l, x_u_w, x_u_m, x_u_s], dim=0))
        logits_l = all_logits[:B]
        logits_w = all_logits[B:B + muB]
        logits_m = all_logits[B + muB:B + 2 * muB]
        logits_s = all_logits[B + 2 * muB:]

        sup_loss = F.cross_entropy(logits_l, y_l)
        tgt_w = F.softmax(logits_w.detach() / 0.5, dim=-1)
        tgt_m = F.softmax(logits_m.detach() / 0.5, dim=-1)
        mask = torch.ones(muB, device=device)

        unsup_loss = (
            (F.cross_entropy(logits_s, logits_w.detach().argmax(-1), reduction="none") * mask).mean()
            + (F.kl_div(F.log_softmax(logits_m, dim=-1), tgt_w, reduction="none").sum(-1) * mask).mean()
            + (F.kl_div(F.log_softmax(logits_s, dim=-1), tgt_m, reduction="none").sum(-1) * mask).mean()
            + (F.kl_div(F.log_softmax(logits_s, dim=-1), tgt_w, reduction="none").sum(-1) * mask).mean()
        )
        loss = sup_loss + unsup_loss
        loss.backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops()


METHODS = {
    "fixmatch": fixmatch_iter_flops,
    "flexmatch": flexmatch_iter_flops,
    "mixmatch": mixmatch_iter_flops,
    "efficientmatch": efficientmatch_iter_flops,
    "efficientmatch_2": efficientmatch_2_iter_flops,
    "efficientmatch_3": efficientmatch_3_iter_flops,
    "efficientmatch_freematch": efficientmatch_freematch_iter_flops,
    "efficientmatch_freematch_svhn": partial(efficientmatch_freematch_iter_flops, svhn_clamp=True),
    "efficientmatch_freematch_c100": partial(efficientmatch_freematch_iter_flops, num_classes=100),
    "efficientmatch_3_mu1": partial(efficientmatch_3_iter_flops, mu=MU["efficientmatch_3_mu1"]),
    "efficientmatch_3_mu5": partial(efficientmatch_3_iter_flops, mu=MU["efficientmatch_3_mu5"]),
    "efficientmatch_3_mu7": partial(efficientmatch_3_iter_flops, mu=MU["efficientmatch_3_mu7"]),
    "efficientmatch_flex_mu2": partial(efficientmatch_iter_flops, mu=MU["efficientmatch_flex_mu2"]),
    "sequencematch": sequencematch_iter_flops,
    "regmixmatch": regmixmatch_iter_flops,
    "regmixmatch_mu3": partial(regmixmatch_iter_flops, mu=MU["regmixmatch_mu3"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=sorted(METHODS), choices=sorted(METHODS))
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen-factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--K", type=int, default=None,
                         help="si fourni, affiche aussi les FLOPs totaux extrapolés sur K itérations")
    parser.add_argument("--out", type=str, default=None, help="chemin du JSON récapitulatif")
    parser.add_argument("--md", type=str, default=None, help="chemin du document Markdown récapitulatif")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} | WideResNet-{args.depth}-{args.widen_factor} | batch labellisé={BATCH_SIZE_L}")

    results = {}
    for name in args.methods:
        model = build_model(device, args.depth, args.widen_factor)
        flops = METHODS[name](model, device)
        mu = MU[name]
        unlabeled_batch = mu * BATCH_SIZE_L
        results[name] = {"mu": mu, "unlabeled_batch": unlabeled_batch, "flops_per_iter": flops}
        line = f"{name:<16} mu={mu}  {flops:>14.3e} FLOPs/it  ({flops / 1e9:.2f} GFLOPs/it)"
        if args.K:
            line += f"  ->  {flops * args.K:.3e} FLOPs sur K={args.K}"
        print(line)

    header = f"{'Méthode':<16}{'mu':>4}{'Batch non labellisé':>22}{'FLOPs/it':>16}{'GFLOPs/it':>14}"
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<16}{r['mu']:>4}{r['unlabeled_batch']:>22}{r['flops_per_iter']:>16.3e}"
              f"{r['flops_per_iter'] / 1e9:>14.2f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=4), encoding="utf-8")
        print(f"\nRésumé JSON écrit dans {args.out}")

    if args.md:
        write_markdown(args.md, args, device, results)
        print(f"Résumé Markdown écrit dans {args.md}")


def write_markdown(out_path, args, device, results):
    lines = [
        "# Résultats de l'analyse FLOPs par itération",
        "",
        f"Mesures réalisées via `scripts/flops_analysis.py` sur {device} "
        f"(`torch=={torch.__version__}`, Python {platform.python_version()}), avec "
        f"`torch.utils.flop_counter.FlopCounterMode` sur des tenseurs factices (dummy tensors) -- "
        f"indépendant des données, de `--optimized`/`--amp`/`torch.compile`.",
        "",
        f"Modèle : WideResNet-{args.depth}-{args.widen_factor} ; batch labellisé = {BATCH_SIZE_L}.",
        "",
        "## Résultats",
        "",
        "| Méthode | mu | Batch non labellisé | FLOPs / itération | GFLOPs / itération |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['mu']} | {r['unlabeled_batch']} | {r['flops_per_iter']:.3e} | "
            f"{r['flops_per_iter'] / 1e9:.2f} |"
        )
    lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
