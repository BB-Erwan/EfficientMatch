"""Utilitaires de chargement de données génériques, partagés par tous les algorithmes SSL. Ne contient
aucune augmentation ni composition de vues (spécifique à chaque algorithme, cf. <algo>.py) -- juste le
split labellisé/non labellisé (doit être identique d'un algo à l'autre pour une comparaison équitable)
et les mécanismes de wrapping/chargement de dataset, génériques par nature."""
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset, RandomSampler


def load_cifar10(data_root):
    train_base = torchvision.datasets.CIFAR10(data_root, train=True, download=True)
    test_base = torchvision.datasets.CIFAR10(data_root, train=False, download=True)
    return train_base, test_base


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
    """Renvoie (image transformée, label)."""

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        return self.transform(img), label


class MultiViewDataset(Dataset):
    """Renvoie un tuple (vue_1, ..., vue_K, label, idx) : K vues augmentées de la même image brute
    (une par transform de `view_transforms` -- la composition des vues est décidée par l'appelant,
    spécifique à chaque algorithme, ex. faible+forte pour FixMatch, K_aug vues faibles indépendantes
    pour MixMatch), son vrai label, et son indice dans ce dataset.

    Le label est chargé ici, via les workers du DataLoader (parallélisé, comme les vues), plutôt que
    recherché après coup dans un tableau séparé à chaque itération -- il n'est JAMAIS utilisé dans la
    perte d'entraînement, uniquement pour le suivi diagnostique de la qualité des pseudo-labels
    (cf. chaque script). L'indice sert au Curriculum Pseudo Labeling de FlexMatch, qui maintient un
    état par échantillon à travers les itérations ; les autres algorithmes l'ignorent."""

    def __init__(self, base_dataset, indices, view_transforms):
        self.base_dataset = base_dataset
        self.indices = indices
        self.view_transforms = view_transforms

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        views = tuple(transform(img) for transform in self.view_transforms)
        return (*views, label, idx)


def infinite_loader(dataset, batch_size, num_workers, shuffle=True):
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
        num_workers=num_workers, drop_last=True,
    )
    while True:
        for batch in loader:
            yield batch
