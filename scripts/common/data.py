"""Datasets, labeled/unlabeled split, augmentations and dataloaders.

Shared by every method because a fair comparison requires them to be identical: the same labeled
subset, drawn the same way from the same seed, and the same weak and strong augmentation pipelines.
What each method *does* with the two views is what differs, and that lives in the method's script.

The order in which these functions consume the global RNG matters. `torch.manual_seed(seed)` is
followed by one `randperm` over the whole training set and then one per class; the model is built
afterwards and draws its initialisation from what is left. Calling them in any other order changes
the labeled subset and the initial weights, and with them every number a run produces.
"""
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset
from PIL import Image
from torchvision.transforms import v2

from datasets_utils import TransformedDataset, TransformedDatasetWithIndex

# Per-dataset channel statistics, used to normalise every view.
DATASETS = {
    "cifar10": dict(num_classes=10, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "cifar100": dict(num_classes=100, mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    "svhn": dict(num_classes=10, mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
    # STL-10: 96x96 images, statistics as in USB (Wang et al., 2022).
    "stl10": dict(num_classes=10, mean=(0.4409, 0.4279, 0.3868), std=(0.2683, 0.2610, 0.2687), size=96),
}


def image_size(name):
    """Side of the (square) images of a dataset: 32 for CIFAR and SVHN, 96 for STL-10."""
    return DATASETS[name].get("size", 32)

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


class LazySTL10(torch.utils.data.Dataset):
    """STL-10 split read through a memory map instead of being loaded into RAM.

    torchvision's STL10 holds the images as one array (2.8 GB for the unlabeled split). On Windows,
    DataLoader workers are spawned and receive a pickled copy of the dataset, which fails at that
    size. This class pickles only the file path and opens the map lazily in each process. Images,
    labels and their order are identical to torchvision.datasets.STL10 (checked image by image).
    """

    def __init__(self, root, split):
        torchvision.datasets.STL10(root=root, split=split, download=True)  # download and verify only
        base = os.path.join(root, "stl10_binary")
        self.path = os.path.join(base, f"{split}_X.bin")
        self.n = os.path.getsize(self.path) // (3 * 96 * 96)
        y = os.path.join(base, f"{split}_y.bin")
        if os.path.exists(y):
            self.labels = (np.fromfile(y, dtype=np.uint8).astype(np.int64) - 1).tolist()
        else:
            self.labels = [-1] * self.n
        self._data = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.n, 3, 96, 96))
        # The binary is column-major: (3, 96, 96) stored as channel, column, row.
        return Image.fromarray(np.ascontiguousarray(self._data[i].transpose(2, 1, 0))), self.labels[i]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None
        return state


def load_datasets(name):
    """(num_classes, mean, std, train_ds, test_ds). Downloads into data/ at the repository root on
    first use, so the scripts can be launched from anywhere."""
    spec = DATASETS[name]
    mean, std = torch.tensor(spec["mean"]), torch.tensor(spec["std"])
    if name == "cifar100":
        train = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True)
        test = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True)
    elif name == "stl10":
        # The 5000 labeled training images and the 100k unlabeled ones, whose label is -1: the labeled
        # subset is drawn from the former, and everything else is unlabeled (see split_labeled_unlabeled).
        train = ConcatDataset([LazySTL10(DATA_ROOT, "train"), LazySTL10(DATA_ROOT, "unlabeled")])
        test = LazySTL10(DATA_ROOT, "test")
    elif name == "svhn":
        train = torchvision.datasets.SVHN(root=DATA_ROOT, split="train", download=True)
        test = torchvision.datasets.SVHN(root=DATA_ROOT, split="test", download=True)
    else:
        train = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
        test = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)
    return spec["num_classes"], mean, std, train, test


def split_labeled_unlabeled(train_ds, num_classes, num_labeled):
    """Draw `num_labeled // num_classes` labeled samples per class, the rest unlabeled.

    Consumes the RNG exactly as described in this module's docstring: one shuffle of the whole set,
    then one permutation per class. Seed the generator before calling.
    """
    train_ds = Subset(train_ds, torch.randperm(len(train_ds)))
    num_per_class = num_labeled // num_classes
    labels = [label for _, label in train_ds]  # read once; the RNG is consumed exactly as before
    labeled_indices, unlabeled_indices = [], []
    for i in range(num_classes):
        class_indices = [j for j, label in enumerate(labels) if label == i]
        perm = torch.randperm(len(class_indices))
        labeled_indices.extend([class_indices[j] for j in perm[:num_per_class]])
        unlabeled_indices.extend([class_indices[j] for j in perm[num_per_class:]])
    # Samples without a label (label -1: STL-10's unlabeled split) are always unlabeled.
    unlabeled_indices.extend([j for j, label in enumerate(labels) if label < 0])
    return Subset(train_ds, labeled_indices), Subset(train_ds, unlabeled_indices)


def class_counts(labeled_ds, num_classes):
    counts = torch.zeros(num_classes)
    for _, label in labeled_ds:
        counts[label] += 1
    return counts


def build_transforms(mean, std, size=32):
    """(normalise-only, weak, strong).

    Weak is the standard flip-and-crop; strong prepends RandAugment. Both are applied on the CPU to
    the whole batch, in the training loop, just before it is moved to the GPU; this is why the
    datasets below are only converted to tensors.

    Given a batch tensor, these v2 transforms draw their random parameters once for the whole batch:
    every image in it gets the same flip, the same crop and the same RandAugment operations. Every
    run of the paper was made this way, for every method alike.

    `size` is the image side (see image_size); the crop padding is size/8, i.e. 4 px at 32 px.
    """
    pad = size // 8
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    weak = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(size, padding=pad),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    strong = v2.Compose([
        v2.RandAugment(num_ops=3, magnitude=5),
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(size, padding=pad),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    return norm, weak, strong


def build_loaders(labeled_ds, unlabeled_ds, test_ds, norm_transform, batch_size_l, mu, optimized):
    """(labeled loader, unlabeled loader, test loader, wrapped unlabeled dataset).

    The unlabeled loader yields each sample's index alongside its image, so the training loop can
    track how that sample's pseudo-label evolves between evaluations. The wrapped unlabeled dataset
    is returned as well because the loop reads the true labels off it to score those pseudo-labels.

    Only ToTensor is applied here: the weak and strong views are built inside the training loop,
    from the same raw batch.
    """
    labeled_ds = TransformedDataset(labeled_ds, transforms.ToTensor())
    unlabeled_ds = TransformedDatasetWithIndex(unlabeled_ds, transform=transforms.ToTensor())
    test_ds = TransformedDataset(test_ds, norm_transform)

    kwargs = dict(num_workers=min(2, os.cpu_count()), pin_memory=True,
                  prefetch_factor=4, persistent_workers=True) if optimized else {}
    return (DataLoader(labeled_ds, batch_size=batch_size_l, shuffle=True, **kwargs),
            DataLoader(unlabeled_ds, batch_size=batch_size_l * mu, shuffle=True, **kwargs),
            DataLoader(test_ds, batch_size=256, shuffle=False, **kwargs),
            unlabeled_ds)
