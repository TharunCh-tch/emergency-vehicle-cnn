"""
Evaluate a trained checkpoint against the held-out test split and write
results.json / results.md with real, measured numbers (accuracy, precision,
recall, F1, ROC-AUC, confusion matrix) plus a description of the run.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import CLASS_NAMES, get_splits
from src.metrics import compute_metrics, roc_points
from src.model import EmergencyVehicleCNN


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[EmergencyVehicleCNN, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = EmergencyVehicleCNN(
        num_classes=2,
        image_size=ckpt.get("image_size", 128),
        dropout=ckpt.get("dropout", 0.4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/raw")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=str, default="results/results.json")
    ap.add_argument("--out-md", type=str, default="results/results.md")
    ap.add_argument("--training-log", type=str, default="results/training_log.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    model, ckpt = load_model(Path(args.checkpoint), device)
    image_size = ckpt.get("image_size", 128)

    _, _, test_ds = get_splits(Path(args.data_root), image_size=image_size, seed=args.seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_true, all_pred, all_score = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_true.extend(y.tolist())
            all_pred.extend(preds.tolist())
            all_score.extend(probs[:, 1].tolist())

    metrics = compute_metrics(all_true, all_pred, all_score)
    fpr, tpr, _ = roc_points(all_true, all_score)

    training_meta = {}
    tl_path = Path(args.training_log)
    if tl_path.exists():
        with open(tl_path, encoding="utf-8") as f:
            training_meta = json.load(f)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "image_size": image_size,
        "test_set_size": len(test_ds),
        "class_names": CLASS_NAMES,
        "metrics": metrics.to_dict(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "device": "cpu",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "train_size": training_meta.get("train_size"),
        "val_size": training_meta.get("val_size"),
        "epochs_trained": len(training_meta.get("history", [])),
        "training_elapsed_seconds": training_meta.get("elapsed_seconds"),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_json}")

    cm = metrics.confusion_matrix
    md = f"""# Evaluation Results

**These are real numbers from an actual run on this machine (CPU only, no
GPU available) — not estimates or numbers copied from the original
Kaggle/GPU coursework project.**

## Run configuration

- Device: CPU (no GPU on this machine)
- Image resolution: {image_size}x{image_size}
- Checkpoint: `{args.checkpoint}` (best val-loss epoch: {ckpt.get('epoch')})
- Train / val / test sizes: {training_meta.get('train_size')} / {training_meta.get('val_size')} / {len(test_ds)}
- Epochs trained: {len(training_meta.get('history', []))}
- Training wall-clock time: {training_meta.get('elapsed_seconds', 0):.1f}s
- PyTorch: {torch.__version__} | Python: {platform.python_version()}

## Test set metrics

| Metric | Value |
|---|---|
| Accuracy | {metrics.accuracy:.4f} |
| Precision | {metrics.precision:.4f} |
| Recall | {metrics.recall:.4f} |
| F1-score | {metrics.f1:.4f} |
| ROC-AUC | {metrics.roc_auc:.4f} |

## Confusion matrix

Rows = true label, columns = predicted label. Class order: `{CLASS_NAMES}`

|  | Pred: non_emergency | Pred: emergency |
|---|---|---|
| True: non_emergency | {cm[0][0]} | {cm[0][1]} |
| True: emergency | {cm[1][0]} | {cm[1][1]} |

## Comparison to the original coursework project

The original LinkedIn-described project reported **81.85% test accuracy**,
trained on a Kaggle dataset using a T4/P100 GPU. This rebuild measured
**{metrics.accuracy*100:.2f}% accuracy** on a much smaller, CPU-feasible
Wikimedia-sourced dataset ({len(test_ds)} test images vs. a presumably much
larger Kaggle set) at {image_size}x{image_size} resolution. See the README
Limitations section for an honest discussion of why these numbers differ and
why that is expected, not a discrepancy to be hidden.
"""
    out_md = Path(args.out_md)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {out_md}")
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
