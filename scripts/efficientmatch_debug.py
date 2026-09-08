"""EfficientMatch -- script de debug, ZÉRO optimisation de vitesse.

Objectif : isoler le comportement de l'algorithme (masque de confiance, canal Mixup) sans qu'aucune
optimisation de vitesse ne puisse interagir avec les résultats. Tout tourne en FP32 standard,
DataLoader synchrone (num_workers=0) : pas d'AMP/bf16, pas de torch.compile, pas de channels_last,
pas de cudnn.benchmark.

Algorithme repris de la référence Semi-SL-trajectories/CIFAR-10/FixMixMatch.ipynb (même méthode :
"FixMixMatch" = "EfficientMatch") -- perte supervisée + cohérence faible/forte filtrée (FixMatch) +
canal Mixup filtré (lambda PAR ÉCHANTILLON, masque du PARTENAIRE MÉLANGÉ, cible SOFT pour la partie
non labellisée, gradient clipping). Le suivi détaillé de la qualité des pseudo-labels (corrections,
reinforcement...) et le mécanisme de checkpoint/reprise de la référence ont été retirés : superflus
pour du debug ponctuel.

Usage :
    python scripts/efficientmatch_debug.py
    python scripts/efficientmatch_debug.py --mu 7 --max-steps 5000
    python scripts/efficientmatch_debug.py --no-grad-clip   # tester l'hypothèse "instabilité sans clipping"
"""
import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset, Subset


# --- Modèle : WideResNet-28-2 (copié de scripts/models.py, sans channels_last ni torch.compile) ---
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_io = in_planes == out_planes and stride == 1
        self.shortcut = None if self.equal_io else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = x if self.equal_io else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=2, drop_rate=0.0):
        super().__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, n_channels[0], 3, stride=1, padding=1, bias=False)
        self.block1 = self._make_block(n_channels[0], n_channels[1], n, stride=1, drop_rate=drop_rate)
        self.block2 = self._make_block(n_channels[1], n_channels[2], n, stride=2, drop_rate=drop_rate)
        self.block3 = self._make_block(n_channels[2], n_channels[3], n, stride=2, drop_rate=drop_rate)
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]

    def _make_block(self, in_planes, out_planes, num_layers, stride, drop_rate):
        layers = [BasicBlock(in_planes, out_planes, stride, drop_rate)]
        for _ in range(1, num_layers):
            layers.append(BasicBlock(out_planes, out_planes, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


# --- Datasets : mêmes wrappers que la référence (TransformedDataset[WithIndex]) ---
class TransformedDataset(Dataset):
    """Applique `transform` à chaque image ; renvoie (image, label)."""

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform(img), label


class TransformedDatasetWithIndex(Dataset):
    """Comme TransformedDataset, mais renvoie aussi l'indice -- pas utilisé dans ce script simplifié,
    gardé pour coller à la structure de la référence."""

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform(img), label, idx


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    """Accuracy simple, sans F1 ni suivi de qualité des pseudo-labels."""
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / total


def make_balanced_split(train_ds_raw, num_labeled, num_classes):
    num_per_class = num_labeled // num_classes
    labeled_indices, unlabeled_indices = [], []
    for c in range(num_classes):
        class_indices = [j for j, (_, label) in enumerate(train_ds_raw) if label == c]
        perm = torch.randperm(len(class_indices))
        labeled_indices.extend([class_indices[j] for j in perm[:num_per_class]])
        unlabeled_indices.extend([class_indices[j] for j in perm[num_per_class:]])
    return Subset(train_ds_raw, labeled_indices), Subset(train_ds_raw, unlabeled_indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-labeled", type=int, default=250)
    parser.add_argument("--mu", type=int, default=3,
                         help="3 dans la référence fixmixmatch, 7 dans le protocole standard du dépôt")
    parser.add_argument("--batch-size-l", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--alpha-mix", type=float, default=0.75)
    parser.add_argument("--max-steps", type=int, default=2000,
                         help="volontairement petit pour un debug rapide (pas 2**17 / 2**20)")
    parser.add_argument("--test-period", type=int, default=100)
    parser.add_argument("--grad-clip", dest="grad_clip", action="store_true", default=True)
    parser.add_argument("--no-grad-clip", dest="grad_clip", action="store_false",
                         help="désactive clip_grad_norm_ -- pour tester l'hypothèse instabilité sans clipping")
    parser.add_argument("--no-plot", action="store_true", help="n'affiche pas les courbes à la fin")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_classes = 10
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Données : CIFAR-10, split labellisé/non labellisé équilibré par classe ---
    train_ds_raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    test_ds_raw = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    print(f"Training samples: {len(train_ds_raw)}, Test samples: {len(test_ds_raw)}")

    train_ds_raw = Subset(train_ds_raw, torch.randperm(len(train_ds_raw)))
    labeled_ds_raw, unlabeled_ds_raw = make_balanced_split(train_ds_raw, args.num_labeled, num_classes)

    labeled_class_counts = torch.zeros(num_classes)
    for _, label in labeled_ds_raw:
        labeled_class_counts[label] += 1
    print(f"Distribution des classes labellisées : {labeled_class_counts.tolist()}")

    # --- Transforms (identiques à la référence) ---
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])
    norm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    weak_transform = v2.Compose([
        v2.ToImage(), v2.RandomHorizontalFlip(), v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std),
    ])
    strong_transform = v2.Compose([
        v2.RandAugment(num_ops=3, magnitude=5),
        v2.ToImage(), v2.RandomHorizontalFlip(), v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std),
    ])

    labeled_ds = TransformedDataset(labeled_ds_raw, transforms.ToTensor())
    unlabeled_ds = TransformedDatasetWithIndex(unlabeled_ds_raw, transforms.ToTensor())
    test_ds = TransformedDataset(test_ds_raw, norm_transform)

    # --- DataLoaders (synchrones, num_workers=0, aucune optimisation) ---
    labeled_loader = DataLoader(labeled_ds, batch_size=args.batch_size_l, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size_l * args.mu, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # --- Modèle, optimiseur, scheduler ---
    model = WideResNet(num_classes=num_classes, depth=28, widen_factor=2).to(device)
    print(f"Nombre de paramètres : {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    beta_dist = torch.distributions.Beta(
        torch.tensor(args.alpha_mix, device=device, dtype=torch.float32),
        torch.tensor(args.alpha_mix, device=device, dtype=torch.float32),
    )

    # --- Boucle d'entraînement EfficientMatch (algo complet, zéro optimisation de vitesse) ---
    # Étapes : (1) pseudo-étiquetage sans gradient sur la vue faible + masque de confiance dur ;
    # (2) un seul forward combiné labellisé+non-labellisé-fort pour perte supervisée + cohérence
    # (comme la référence -- le forward séparé de scripts/algorithms/fixmatch.py donnerait des
    # statistiques BatchNorm différentes, à garder en tête si vous comparez aux runs du dépôt) ;
    # (3) canal Mixup avec lambda par échantillon, masque du partenaire mélangé, cible soft pour la
    # partie non labellisée ; (4) clip_grad_norm_ optionnel (--no-grad-clip pour le désactiver).
    history = {"step": [], "loss": [], "loss_s": [], "loss_u": [], "loss_mix": [], "mask_ratio": []}
    eval_history = {"step": [], "acc": []}

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    mask_ratio_buffer, loss_buffer = [], []
    start_time = time.time()

    for step in range(args.max_steps):
        model.train()

        try:
            x_l, y_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x_l, y_l = next(labeled_iter)

        try:
            x_u, _, idx = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_u, _, idx = next(unlabeled_iter)

        x_l_w = weak_transform(x_l).to(device)
        y_l = y_l.to(device)
        x_u_w = weak_transform(x_u).to(device)
        x_u_s = strong_transform(x_u).to(device)

        # --- Pseudo-étiquetage (sans gradient, FP32) ---
        with torch.no_grad():
            logits_u_w = model(x_u_w)
            probs_u_w = F.softmax(logits_u_w, dim=1)
            max_prob, pseudo = torch.max(probs_u_w, dim=1)
            mask = max_prob.ge(args.tau).float()
            mask_ratio_buffer.append(mask.mean().item())

        optimizer.zero_grad()

        # --- Forward fusionné supervisé + cohérence (comme la référence) ---
        B = x_l_w.shape[0]
        all_logits = model(torch.cat([x_l_w, x_u_s], dim=0))
        logits_l = all_logits[:B]
        logits_u_s = all_logits[B:]

        loss_s = F.cross_entropy(logits_l, y_l)
        loss_u = (mask * F.cross_entropy(logits_u_s, pseudo, reduction="none")).mean()

        # --- Canal Mixup filtré (lambda par échantillon, masque partenaire, cible soft côté u) ---
        all_imgs = torch.cat([x_l_w, x_u_w], dim=0)
        all_labels = torch.cat(
            [F.one_hot(y_l, num_classes).float(), F.one_hot(pseudo, num_classes).float()], dim=0
        )
        all_masks = torch.cat([torch.ones(B, device=device), mask], dim=0)

        perm = torch.randperm(all_imgs.size(0), device=device)
        all_imgs_shuffled = all_imgs[perm]
        all_labels_shuffled = all_labels[perm]
        all_masks_shuffled = all_masks[perm]

        lam_x = beta_dist.sample((B,))
        lam_u = beta_dist.sample((x_u_w.size(0),))
        lam_x = torch.maximum(lam_x, 1 - lam_x)
        lam_u = torch.maximum(lam_u, 1 - lam_u)

        mixed_x_imgs = torch.lerp(all_imgs_shuffled[:B], x_l_w, lam_x.view(-1, 1, 1, 1))
        mixed_u_imgs = torch.lerp(all_imgs_shuffled[B:], x_u_w, lam_u.view(-1, 1, 1, 1))

        mixed_x_labels = torch.lerp(all_labels_shuffled[:B], F.one_hot(y_l, num_classes).float(), lam_x.view(-1, 1))
        mixed_u_labels = torch.lerp(all_labels_shuffled[B:], probs_u_w, lam_u.view(-1, 1))

        logits_mixed = model(torch.cat([mixed_x_imgs, mixed_u_imgs], dim=0))
        logits_x_mixed = logits_mixed[:B]
        logits_u_mixed = logits_mixed[B:]

        mask_x_shuffled = all_masks_shuffled[:B]
        mask_u_shuffled = all_masks_shuffled[B:]

        loss_mix = (
            (mask_x_shuffled * F.cross_entropy(logits_x_mixed, mixed_x_labels, reduction="none")).mean()
            + (mask_u_shuffled * F.cross_entropy(logits_u_mixed, mixed_u_labels, reduction="none")).mean()
        )

        loss = loss_s + loss_u + loss_mix
        loss_buffer.append(loss.item())

        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % args.test_period == 0 or step == 0 or step == args.max_steps - 1:
            acc = evaluate_accuracy(model, test_loader, device)
            history["step"].append(step + 1)
            history["loss"].append(np.mean(loss_buffer))
            history["loss_s"].append(loss_s.item())
            history["loss_u"].append(loss_u.item())
            history["loss_mix"].append(loss_mix.item())
            history["mask_ratio"].append(np.mean(mask_ratio_buffer))
            eval_history["step"].append(step + 1)
            eval_history["acc"].append(acc)
            loss_buffer, mask_ratio_buffer = [], []

            elapsed = time.time() - start_time
            print(
                f"[step {step + 1:>5}/{args.max_steps}] loss={history['loss'][-1]:.4f} "
                f"(sup={loss_s.item():.4f} cons={loss_u.item():.4f} mix={loss_mix.item():.4f}) "
                f"mask_ratio={history['mask_ratio'][-1]:.4f} acc={acc:.4f} elapsed={elapsed:.1f}s"
            )

    if not args.no_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history["step"], history["loss"], label="totale")
        axes[0].plot(history["step"], history["loss_s"], label="supervisée", alpha=0.6)
        axes[0].plot(history["step"], history["loss_u"], label="cohérence", alpha=0.6)
        axes[0].plot(history["step"], history["loss_mix"], label="mixup", alpha=0.6)
        axes[0].set_title("Perte")
        axes[0].set_xlabel("itération")
        axes[0].legend()

        axes[1].plot(history["step"], history["mask_ratio"], color="tab:orange")
        axes[1].set_title(f"Taux de masque de confiance (tau={args.tau})")
        axes[1].set_xlabel("itération")
        axes[1].set_ylim(0, 1)

        axes[2].plot(eval_history["step"], eval_history["acc"], color="tab:green")
        axes[2].set_title("Accuracy (test)")
        axes[2].set_xlabel("itération")
        axes[2].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
