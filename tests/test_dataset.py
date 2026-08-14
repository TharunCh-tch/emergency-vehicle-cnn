"""Tests for the data pipeline: split proportions and transform correctness.

Uses small synthetic in-memory images/fixtures (written to a temp dir) so
tests run fast without needing the real scraped dataset.
"""
from pathlib import Path

import torch
from PIL import Image

from src.dataset import Sample, build_transforms, list_samples, split_samples


def make_fake_dataset(tmp_path: Path, n_emergency: int = 40, n_non: int = 40) -> Path:
    raw_root = tmp_path / "raw"
    for class_name, n in [("emergency", n_emergency), ("non_emergency", n_non)]:
        class_dir = raw_root / class_name
        class_dir.mkdir(parents=True)
        for i in range(n):
            img = Image.new("RGB", (32, 32), color=(i % 256, 100, 150))
            img.save(class_dir / f"{class_name}_{i:03d}.jpg")
    return raw_root


def test_list_samples_finds_all_images(tmp_path):
    raw_root = make_fake_dataset(tmp_path, n_emergency=10, n_non=15)
    samples = list_samples(raw_root)
    assert len(samples) == 25
    labels = [s.label for s in samples]
    assert labels.count(1) == 10  # emergency == 1
    assert labels.count(0) == 15  # non_emergency == 0


def test_split_proportions_approximately_70_15_15():
    samples = [Sample(path=Path(f"f{i}.jpg"), label=i % 2) for i in range(200)]
    train, val, test = split_samples(samples, train_frac=0.7, val_frac=0.15, seed=42)

    total = len(train) + len(val) + len(test)
    assert total == len(samples)
    assert abs(len(train) / total - 0.7) < 0.03
    assert abs(len(val) / total - 0.15) < 0.03
    assert abs(len(test) / total - 0.15) < 0.03


def test_split_is_disjoint():
    samples = [Sample(path=Path(f"f{i}.jpg"), label=i % 2) for i in range(100)]
    train, val, test = split_samples(samples, seed=1)
    train_paths = {s.path for s in train}
    val_paths = {s.path for s in val}
    test_paths = {s.path for s in test}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_split_is_deterministic_given_seed():
    samples = [Sample(path=Path(f"f{i}.jpg"), label=i % 2) for i in range(60)]
    train1, val1, test1 = split_samples(samples, seed=7)
    train2, val2, test2 = split_samples(samples, seed=7)
    assert [s.path for s in train1] == [s.path for s in train2]
    assert [s.path for s in val1] == [s.path for s in val2]
    assert [s.path for s in test1] == [s.path for s in test2]


def test_split_preserves_class_balance_per_split():
    # 60 emergency + 60 non_emergency -> each split should stay balanced
    samples = [Sample(path=Path(f"e{i}.jpg"), label=1) for i in range(60)]
    samples += [Sample(path=Path(f"n{i}.jpg"), label=0) for i in range(60)]
    train, val, test = split_samples(samples, seed=3)
    for split in (train, val, test):
        n_pos = sum(1 for s in split if s.label == 1)
        n_neg = sum(1 for s in split if s.label == 0)
        assert abs(n_pos - n_neg) <= 1


def test_transform_output_shape_and_type():
    transform = build_transforms(image_size=96, train=False)
    img = Image.new("RGB", (200, 150), color=(10, 20, 30))
    out = transform(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 96, 96)
    assert out.dtype == torch.float32


def test_transform_train_vs_eval_both_produce_correct_shape():
    img = Image.new("RGB", (64, 64), color=(5, 5, 5))
    train_t = build_transforms(image_size=128, train=True)
    eval_t = build_transforms(image_size=128, train=False)
    assert train_t(img).shape == (3, 128, 128)
    assert eval_t(img).shape == (3, 128, 128)
