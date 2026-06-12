from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


def make_fake_classification(
    size: int,
    input_shape: Sequence[int],
    num_classes: int,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    targets = torch.arange(size) % num_classes
    targets = targets[torch.randperm(size, generator=generator)]
    prototypes = torch.randn((num_classes, *input_shape), generator=generator)
    inputs = prototypes[targets] + 0.5 * torch.randn(
        (size, *input_shape),
        generator=generator,
    )
    return TensorDataset(inputs, targets)


def classification_dataset(
    name: str,
    root: str,
    train: bool,
    download: bool,
    seed: int,
    fake_size: int = 2_000,
    augmentation: bool = False,
) -> tuple[Dataset, tuple[int, ...], int]:
    normalized = name.lower()
    if normalized in {"fake", "fake_mnist"}:
        shape = (1, 28, 28)
        return make_fake_classification(fake_size, shape, 10, seed), shape, 10
    if normalized == "fake_cifar":
        shape = (3, 32, 32)
        return make_fake_classification(fake_size, shape, 10, seed), shape, 10
    if normalized == "mnist":
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root, train=train, transform=transform, download=download)
        return dataset, (1, 28, 28), 10
    if normalized == "cifar10":
        if train and augmentation:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            transform = transforms.ToTensor()
        dataset = datasets.CIFAR10(root, train=train, transform=transform, download=download)
        return dataset, (3, 32, 32), 10
    if normalized == "cifar100":
        transform = transforms.ToTensor()
        dataset = datasets.CIFAR100(root, train=train, transform=transform, download=download)
        return dataset, (3, 32, 32), 100
    if normalized == "imagefolder":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        split = "train" if train else "val"
        dataset = datasets.ImageFolder(str(Path(root) / split), transform=transform)
        return dataset, (3, 224, 224), len(dataset.classes)
    raise ValueError(f"Unknown dataset: {name}")


def dataset_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent = dataset_targets(dataset.dataset)
        return parent[np.asarray(dataset.indices)]
    if hasattr(dataset, "targets"):
        return np.asarray(getattr(dataset, "targets"))
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].detach().cpu().numpy()
    return np.asarray([int(dataset[index][1]) for index in range(len(dataset))])


def balanced_subset(dataset: Dataset, size: int, seed: int) -> Subset:
    if size >= len(dataset):
        return Subset(dataset, list(range(len(dataset))))
    targets = dataset_targets(dataset)
    classes = np.unique(targets)
    per_class = size // len(classes)
    remainder = size % len(classes)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for offset, class_id in enumerate(classes):
        candidates = np.flatnonzero(targets == class_id)
        count = per_class + int(offset < remainder)
        if count > len(candidates):
            raise ValueError(f"Class {class_id} has only {len(candidates)} examples")
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected)


def batch_from_indices(
    dataset: Dataset,
    indices: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    examples = [dataset[int(index)] for index in indices]
    inputs = torch.stack([example[0] for example in examples]).to(device)
    targets = torch.as_tensor([int(example[1]) for example in examples], device=device)
    return inputs, targets


class PermutedDataset(Dataset):
    def __init__(self, base: Dataset, permutation: torch.Tensor) -> None:
        self.base = base
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        inputs, target = self.base[index]
        flattened = inputs.flatten()[self.permutation]
        return flattened.reshape_as(inputs), int(target)


class RemappedSubset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int], class_offset: int) -> None:
        self.base = base
        self.indices = list(int(value) for value in indices)
        self.class_offset = class_offset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        inputs, target = self.base[self.indices[index]]
        return inputs, int(target) - self.class_offset


def split_class_tasks(
    dataset: Dataset,
    classes_per_task: int,
    task_count: int,
) -> list[Dataset]:
    targets = dataset_targets(dataset)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    tasks: list[Dataset] = []
    for task_id in range(task_count):
        first_class = task_id * classes_per_task
        indices: list[int] = []
        for class_id in range(first_class, first_class + classes_per_task):
            indices.extend(by_class[class_id])
        tasks.append(RemappedSubset(dataset, indices, first_class))
    return tasks
