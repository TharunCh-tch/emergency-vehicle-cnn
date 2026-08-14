"""
Dataset loading and train/val/test splitting for the emergency vehicle
classifier.

Images are read from data/raw/<class>/*.{jpg,png} where <class> is either
"emergency" or "non_emergency". A deterministic 70/15/15 split (by seeded
shuffle) is applied per class so that class balance is preserved across
splits.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["non_emergency", "emergency"]  # index 0 / 1 -> label
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class Sample:
    path: Path
    label: int


def list_samples(raw_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = raw_root / class_name
        if not class_dir.exists():
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append(Sample(path=p, label=label))
    return samples


def split_samples(
    samples: list[Sample],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Deterministic per-class 70/15/15 split (train/val/test)."""
    rng = random.Random(seed)
    by_class: dict[int, list[Sample]] = {}
    for s in samples:
        by_class.setdefault(s.label, []).append(s)

    train, val, test = [], [], []
    for label, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class VehicleDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int = 128, train: bool = False):
        self.samples = samples
        self.transform = build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("RGB")
        img = self.transform(img)
        return img, sample.label


def get_splits(
    raw_root: Path,
    image_size: int = 128,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[VehicleDataset, VehicleDataset, VehicleDataset]:
    samples = list_samples(raw_root)
    train_s, val_s, test_s = split_samples(samples, train_frac, val_frac, seed)
    train_ds = VehicleDataset(train_s, image_size=image_size, train=True)
    val_ds = VehicleDataset(val_s, image_size=image_size, train=False)
    test_ds = VehicleDataset(test_s, image_size=image_size, train=False)
    return train_ds, val_ds, test_ds
