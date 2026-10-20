from typing import Tuple, List
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms


def _extract_targets(dataset: torch.utils.data.Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        sub_targets = _extract_targets(dataset.dataset)
        return sub_targets[np.array(dataset.indices)]

    if hasattr(dataset, "targets") and dataset.targets is not None:
        targets = dataset.targets
    elif hasattr(dataset, "labels") and dataset.labels is not None:
        targets = dataset.labels
    elif hasattr(dataset, "_labels") and dataset._labels is not None:
        targets = dataset._labels
    elif hasattr(dataset, "_targets") and dataset._targets is not None:
        targets = dataset._targets
    else:
        targets = [y for _, y in dataset]

    if isinstance(targets, torch.Tensor):
        return targets.cpu().numpy()
    return np.array(targets)


def partition_dataset_dirichlet(
    dataset: torch.utils.data.Dataset,
    num_clients: int,
    alpha: float = 0.5,
    num_classes: int = 10,
    seed: int = 42
) -> List[Subset]:
    """Particionado Dirichlet determinista con semilla fija."""
    rng = np.random.default_rng(seed)
    targets = _extract_targets(dataset)

    idx_by_class = [np.where(targets == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue

        idx_c = idx_c.copy()
        rng.shuffle(idx_c)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions / proportions.sum())
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]

        idx_splits = np.split(idx_c, splits)
        for i in range(num_clients):
            client_indices[i].extend(idx_splits[i])

    return [Subset(dataset, idxs) for idxs in client_indices]


def calculate_entropy(subset: Subset, num_classes: int = 10) -> float:
    """Calcula H(D_i) en milisegundos directamente sobre el vector de targets."""
    targets = _extract_targets(subset)
    if len(targets) == 0:
        return 0.0
    counts = np.bincount(targets, minlength=num_classes)
    probs = counts / max(1, len(targets))
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


class DatasetFactory:

    @staticmethod
    def get_dataset(
        dataset_name: str = "cifar10",
        batch_size: int = 32,
        max_train_samples: int = 2400,
        max_test_samples: int = 500,
        resolution: Tuple[int, int] = (128, 128),
        client_id: int = 0,
        num_clients: int = 1,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, int]:
        ds = dataset_name.lower().strip()

        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        transform_train = transforms.Compose([
            transforms.Resize(resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

        transform_test = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

        if ds == "mnist":
            transform_mnist = transforms.Compose([
                transforms.Resize(resolution),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_mnist)
            test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_mnist)
            num_classes = 10

        elif ds == "svhn":
            train_set = torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=transform_train)
            test_set = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=transform_test)
            num_classes = 10

        elif ds == "cifar10":
            train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
            test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
            num_classes = 10

        elif ds == "eurosat":
            full_set = torchvision.datasets.EuroSAT(root="./data", download=True, transform=transform_train)
            num_classes = 10
            train_size = int(0.8 * len(full_set))
            test_size = len(full_set) - train_size
            train_set, test_set = random_split(
                full_set, [train_size, test_size], generator=torch.Generator().manual_seed(42)
            )

        elif ds == "gtsrb":
            train_set = torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=transform_train)
            test_set = torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=transform_test)
            num_classes = 43

        elif ds in ["plantvillage", "plant_village"]:
            candidate_paths = [
                "./data/color",
                "./data/plantvillage/raw/color",
                "./data/PlantVillage-Dataset/raw/color",
                "./data/plantvillage",
            ]
            data_dir = next((p for p in candidate_paths if os.path.exists(p) and os.listdir(p)), None)
            if data_dir is None:
                raise FileNotFoundError("No se encontró la carpeta ./data/color de PlantVillage.")

            full_set = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_train)
            num_classes = len(full_set.classes)
            train_size = int(0.8 * len(full_set))
            test_size = len(full_set) - train_size
            train_set, test_set = random_split(
                full_set, [train_size, test_size], generator=torch.Generator().manual_seed(42)
            )

        elif ds in ["pets", "oxfordpet"]:
            train_set = torchvision.datasets.OxfordIIITPet(root="./data", split="trainval", download=True, transform=transform_train)
            test_set = torchvision.datasets.OxfordIIITPet(root="./data", split="test", download=True, transform=transform_test)
            num_classes = 37

        elif ds == "cifar100":
            train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
            test_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
            num_classes = 100

        elif ds in ["flowers102", "flowers"]:
            train_set = torchvision.datasets.Flowers102(root="./data", split="train", download=True, transform=transform_train)
            test_set = torchvision.datasets.Flowers102(root="./data", split="test", download=True, transform=transform_test)
            num_classes = 102

        else:
            raise ValueError(f"Dataset no soportado: '{dataset_name}'")

        alpha = kwargs.get("alpha", 0.5)
        seed = kwargs.get("seed", 42)

        if num_clients > 1:
            client_subsets = partition_dataset_dirichlet(
                dataset=train_set,
                num_clients=num_clients,
                alpha=alpha,
                num_classes=num_classes,
                seed=seed
            )
            raw_client_subset = client_subsets[client_id]

            if max_train_samples and len(raw_client_subset) > max_train_samples:
                g_c = torch.Generator().manual_seed(seed + client_id)
                perm = torch.randperm(len(raw_client_subset), generator=g_c)[:max_train_samples].tolist()
                client_train_subset = Subset(raw_client_subset, perm)
            else:
                client_train_subset = raw_client_subset
        else:
            g = torch.Generator().manual_seed(seed)
            train_len = len(train_set)
            num_train = min(train_len, max_train_samples)
            train_indices = torch.randperm(train_len, generator=g)[:num_train].tolist()
            client_train_subset = Subset(train_set, train_indices)

        g_test = torch.Generator().manual_seed(seed)
        test_len = len(test_set)
        num_test = min(test_len, max_test_samples)
        test_indices = torch.randperm(test_len, generator=g_test)[:num_test].tolist()
        client_test_subset = Subset(test_set, test_indices)

        train_loader = DataLoader(client_train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(client_test_subset, batch_size=64, shuffle=False, num_workers=0)

        return train_loader, test_loader, num_classes
