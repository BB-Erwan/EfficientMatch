"""Chargement des données CIFAR-10/100/PathMNIST et augmentations pour l'entraînement semi-supervisé.

Bascule transparente entre torchvision.transforms (v1, image par image) et
torchvision.transforms.v2 (batch vectorisé) via cfg["use_transforms_v2"].
"""
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms_v1
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, Dataset, Subset

# Résolution d'image et normalisation par canal, propres à chaque dataset.
# CIFAR-10/100 : statistiques standard du protocole FixMatch/FlexMatch.
# PathMNIST : images natives 28x28 (medmnist ne fournit pas de variante 32x32 dans toutes les
# versions du package) ; normalisation (0.5, 0.5, 0.5) = convention utilisée dans le code officiel
# MedMNIST (https://github.com/MedMNIST/MedMNIST), pas une statistique recalculée ici.
DATASET_META = {
    "cifar10":   {"image_size": 32, "mean": (0.4914, 0.4822, 0.4465), "std": (0.2471, 0.2435, 0.2616)},
    "cifar100":  {"image_size": 32, "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762)},
    "pathmnist": {"image_size": 28, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
}


def build_transforms(cfg):
    """Retourne (weak_transform, strong_transform, eval_transform) selon cfg["use_transforms_v2"]
    et les statistiques (taille, moyenne, écart-type) du dataset choisi (cf. DATASET_META)."""
    if cfg["dataset"] not in DATASET_META:
        raise ValueError(f"Dataset inconnu : {cfg['dataset']} (choix : {sorted(DATASET_META)})")
    meta = DATASET_META[cfg["dataset"]]
    size, mean, std = meta["image_size"], meta["mean"], meta["std"]

    use_v2 = cfg["use_transforms_v2"]
    T = transforms_v2 if use_v2 else transforms_v1

    if use_v2:
        # v2 : un seul appel vectorisé sur tout le batch (CPU ou GPU), plus de boucle Python par image
        weak_transform = T.Compose([
            T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4, padding_mode="reflect"),
            T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std),
        ])
        strong_transform = T.Compose([
            T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4, padding_mode="reflect"),
            T.RandAugment(num_ops=2, magnitude=10),
            T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std), T.RandomErasing(p=0.5),
        ])
        eval_transform = T.Compose([
            T.PILToTensor(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std),
        ])
    else:
        # v1 (classique) : transform appliqué image par image (boucle Python) dans la boucle d'entraînement
        weak_transform = T.Compose([
            T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4, padding_mode="reflect"),
            T.ToTensor(), T.Normalize(mean, std),
        ])
        strong_transform = T.Compose([
            T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4, padding_mode="reflect"),
            T.RandAugment(num_ops=2, magnitude=10),
            T.ToTensor(), T.Normalize(mean, std), T.RandomErasing(p=0.5),
        ])
        eval_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    return weak_transform, strong_transform, eval_transform


class BatchAugmenter:
    """Applique un transform à un batch, en basculant v1 (liste de PIL, boucle Python) / v2 (tenseur vectorisé)."""

    def __init__(self, device, use_transforms_v2):
        self.device = device
        self.use_v2 = use_transforms_v2

    def __call__(self, transform, raw_batch):
        if self.use_v2:
            return transform(raw_batch.to(self.device, non_blocking=True))
        return torch.stack([transform(img) for img in raw_batch]).to(self.device, non_blocking=True)


class SSLCollate:
    """Collate custom : le DataLoader ne sait pas empiler nativement une liste d'images PIL (mode v1)."""

    def __init__(self, use_transforms_v2):
        self.use_v2 = use_transforms_v2

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs) if self.use_v2 else list(imgs)
        return imgs, torch.tensor(labels)


class SSLDataset(Dataset):
    def __init__(self, base_dataset, indices, use_transforms_v2):
        self.base_dataset = base_dataset
        self.indices = indices
        self.use_v2 = use_transforms_v2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        if hasattr(label, "item"):  # medmnist renvoie un ndarray de forme (1,) plutôt qu'un int
            label = int(np.asarray(label).reshape(-1)[0])
        if self.use_v2:
            img = transforms_v2.functional.pil_to_tensor(img)  # uint8 CHW -> collate direct en batch tenseur
        return img, label


def make_ssl_split(targets, n_labels, num_classes, seed=0):
    """Split équilibré par classe : n_labels au total, répartis également entre classes.
    `targets` est un vecteur d'entiers (indépendant du dataset d'origine)."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets).reshape(-1)
    n_per_class = n_labels // num_classes
    labeled_idx = []
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        labeled_idx.extend(idx_c[:n_per_class].tolist())
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.arange(len(targets))
    return labeled_idx, unlabeled_idx


def _load_pathmnist(cfg, eval_transform):
    try:
        from medmnist import PathMNIST
    except ImportError as e:
        raise ImportError(
            "Le dataset 'pathmnist' nécessite le package `medmnist` (pip install medmnist, "
            "cf. scripts/requirements.txt)."
        ) from e
    # Contrairement à torchvision.datasets, medmnist ne crée pas `root` automatiquement.
    Path(cfg["data_root"]).mkdir(parents=True, exist_ok=True)
    # medmnist renvoie un label comme ndarray de forme (1,), pas un int : sans target_transform, le
    # DataLoader de test (collate par défaut, pas de passage par SSLDataset) empilerait les labels en
    # (N, 1) et `preds == labels` ferait un broadcast (N, N) silencieusement faux dans evaluate().
    squeeze_label = lambda y: int(np.asarray(y).reshape(-1)[0])  # noqa: E731
    # Résolution native (28x28) : pas de `size=` explicite pour rester compatible avec toutes les
    # versions du package medmnist (seules certaines versions récentes exposent size=32/64/128/224).
    train_base = PathMNIST(root=cfg["data_root"], split="train", download=True, target_transform=squeeze_label)
    test_base = PathMNIST(
        root=cfg["data_root"], split="test", download=True,
        transform=eval_transform, target_transform=squeeze_label,
    )
    targets = np.asarray(train_base.labels).reshape(-1)  # attribut brut, indépendant de target_transform
    return train_base, test_base, targets


def load_datasets(cfg, eval_transform):
    if cfg["dataset"] == "cifar10":
        train_base = torchvision.datasets.CIFAR10(cfg["data_root"], train=True, download=True)
        test_base = torchvision.datasets.CIFAR10(cfg["data_root"], train=False, download=True, transform=eval_transform)
        targets = np.array(train_base.targets)
    elif cfg["dataset"] == "cifar100":
        train_base = torchvision.datasets.CIFAR100(cfg["data_root"], train=True, download=True)
        test_base = torchvision.datasets.CIFAR100(cfg["data_root"], train=False, download=True, transform=eval_transform)
        targets = np.array(train_base.targets)
    elif cfg["dataset"] == "pathmnist":
        train_base, test_base, targets = _load_pathmnist(cfg, eval_transform)
    else:
        raise ValueError(f"Dataset inconnu : {cfg['dataset']} (choix : {sorted(DATASET_META)})")

    labeled_idx, unlabeled_idx = make_ssl_split(targets, cfg["n_labels"], cfg["num_classes"], seed=cfg["seed"])
    if cfg["debug_subset_size"] is not None:
        unlabeled_idx = unlabeled_idx[: cfg["debug_subset_size"]]
        test_base = Subset(test_base, list(range(min(len(test_base), cfg["debug_subset_size"]))))

    labeled_set = SSLDataset(train_base, labeled_idx, cfg["use_transforms_v2"])
    unlabeled_set = SSLDataset(train_base, unlabeled_idx, cfg["use_transforms_v2"])
    return labeled_set, unlabeled_set, test_base


def infinite_loader(dataset, batch_size, cfg, collate_fn, shuffle=True):
    """Itérateur infini sur un DataLoader (labellisé/non-labellisé n'ont pas la même taille d'époque)."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["persistent_workers"] and cfg["num_workers"] > 0, drop_last=True,
        collate_fn=collate_fn,
    )
    while True:
        for batch in loader:
            yield batch
