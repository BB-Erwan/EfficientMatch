"""Chargement des données CIFAR-10/100/PathMNIST et augmentations pour l'entraînement semi-supervisé."""
from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

# Résolution d'image et normalisation par canal, propres à chaque dataset.
# CIFAR-10/100 : statistiques standard du protocole FixMatch.
# PathMNIST : images natives 28x28 (medmnist ne fournit pas de variante 32x32 dans toutes les
# versions du package) ; normalisation (0.5, 0.5, 0.5) = convention utilisée dans le code officiel
# MedMNIST (https://github.com/MedMNIST/MedMNIST), pas une statistique recalculée ici.
DATASET_META = {
    "cifar10":   {"image_size": 32, "mean": (0.4914, 0.4822, 0.4465), "std": (0.2471, 0.2435, 0.2616)},
    "cifar100":  {"image_size": 32, "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762)},
    "pathmnist": {"image_size": 28, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
}


def build_transforms(cfg):
    """Retourne (weak_transform, strong_transform, eval_transform) selon les statistiques
    (taille, moyenne, écart-type) du dataset choisi (cf. DATASET_META)."""
    if cfg["dataset"] not in DATASET_META:
        raise ValueError(f"Dataset inconnu : {cfg['dataset']} (choix : {sorted(DATASET_META)})")
    meta = DATASET_META[cfg["dataset"]]
    size, mean, std = meta["image_size"], meta["mean"], meta["std"]

    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(size, padding=4, padding_mode="reflect"),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    strong_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(size, padding=4, padding_mode="reflect"),
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.ToTensor(), transforms.Normalize(mean, std), transforms.RandomErasing(p=0.5),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    return weak_transform, strong_transform, eval_transform


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


class LabeledDataset(Dataset):
    """Renvoie (image transformée, label) pour le jeu labellisé."""

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        return self.transform(img), label


class UnlabeledDataset(Dataset):
    """Renvoie (vue faible, vue forte) de la même image brute, pour le jeu non labellisé."""

    def __init__(self, base_dataset, indices, weak_transform, strong_transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[self.indices[idx]]
        return self.weak_transform(img), self.strong_transform(img)


def _load_pathmnist(cfg):
    try:
        from medmnist import PathMNIST
    except ImportError as e:
        raise ImportError(
            "Le dataset 'pathmnist' nécessite le package `medmnist` (pip install medmnist, "
            "cf. scripts/requirements.txt)."
        ) from e
    # Contrairement à torchvision.datasets, medmnist ne crée pas `root` automatiquement.
    Path(cfg["data_root"]).mkdir(parents=True, exist_ok=True)
    # medmnist renvoie un label comme ndarray de forme (1,), pas un int.
    squeeze_label = lambda y: int(np.asarray(y).reshape(-1)[0])  # noqa: E731
    # Résolution native (28x28) : pas de `size=` explicite pour rester compatible avec toutes les
    # versions du package medmnist (seules certaines versions récentes exposent size=32/64/128/224).
    train_base = PathMNIST(root=cfg["data_root"], split="train", download=True, target_transform=squeeze_label)
    test_base = PathMNIST(root=cfg["data_root"], split="test", download=True, target_transform=squeeze_label)
    targets = np.asarray(train_base.labels).reshape(-1)  # attribut brut, indépendant de target_transform
    return train_base, test_base, targets


def load_datasets(cfg, weak_transform, strong_transform, eval_transform):
    if cfg["dataset"] == "cifar10":
        train_base = torchvision.datasets.CIFAR10(cfg["data_root"], train=True, download=True)
        test_base = torchvision.datasets.CIFAR10(cfg["data_root"], train=False, download=True)
        targets = np.array(train_base.targets)
    elif cfg["dataset"] == "cifar100":
        train_base = torchvision.datasets.CIFAR100(cfg["data_root"], train=True, download=True)
        test_base = torchvision.datasets.CIFAR100(cfg["data_root"], train=False, download=True)
        targets = np.array(train_base.targets)
    elif cfg["dataset"] == "pathmnist":
        train_base, test_base, targets = _load_pathmnist(cfg)
    else:
        raise ValueError(f"Dataset inconnu : {cfg['dataset']} (choix : {sorted(DATASET_META)})")

    labeled_idx, unlabeled_idx = make_ssl_split(targets, cfg["n_labels"], cfg["num_classes"], seed=cfg["seed"])
    if cfg["debug_subset_size"] is not None:
        unlabeled_idx = unlabeled_idx[: cfg["debug_subset_size"]]
        test_base = Subset(test_base, list(range(min(len(test_base), cfg["debug_subset_size"]))))

    labeled_set = LabeledDataset(train_base, labeled_idx, weak_transform)
    unlabeled_set = UnlabeledDataset(train_base, unlabeled_idx, weak_transform, strong_transform)
    test_set = LabeledDataset(test_base, list(range(len(test_base))), eval_transform)
    return labeled_set, unlabeled_set, test_set


def infinite_loader(dataset, batch_size, cfg, shuffle=True):
    """Itérateur infini sur un DataLoader (labellisé/non-labellisé n'ont pas la même taille d'époque).

    Utilise un RandomSampler AVEC REMISE (pratique standard en SSL) plutôt que `shuffle=True` seul :
    en régime de faible labellisation, le jeu labellisé (n_labels, ex. 250) est plus petit que le
    batch (B, ex. 64) une fois multiplié par le nombre de classes. Avec `shuffle=True` +
    `drop_last=True`, le sampler par défaut tire exactement len(dataset) indices SANS remise par
    "époque" -- si len(dataset) < batch_size, aucun batch complet ne peut jamais être formé, et
    `while True: for batch in loader` boucle indéfiniment sans jamais rien produire (blocage
    silencieux, sans erreur). L'échantillonnage avec remise cycle sur le jeu labellisé autant de fois
    que nécessaire, quelle que soit sa taille par rapport à B.
    """
    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size * 100) if shuffle else None
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False if sampler else shuffle,
        num_workers=cfg["num_workers"], drop_last=True,
    )
    while True:
        for batch in loader:
            yield batch
