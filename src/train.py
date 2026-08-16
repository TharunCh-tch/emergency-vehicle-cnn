"""
Training loop for the EmergencyVehicleCNN with checkpointing and per-epoch
validation tracking. CPU-only (no GPU on this machine).

Usage:
    python -m src.train --epochs 20 --batch-size 32 --image-size 128
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import get_splits
from src.metrics import compute_metrics
from src.model import EmergencyVehicleCNN


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_true, all_pred, all_score = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_score.extend(probs[:, 1].cpu().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = compute_metrics(all_true, all_pred, all_score)
    return avg_loss, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/raw")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--log-path", type=str, default="results/training_log.json")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--select-by",
        type=str,
        default="val_f1",
        choices=["val_loss", "val_f1", "val_acc"],
        help="metric used to pick the checkpoint saved as best_model.pt",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = get_splits(
        Path(args.data_root), image_size=args.image_size, seed=args.seed
    )
    print(f"Split sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = EmergencyVehicleCNN(
        num_classes=2,
        image_size=args.image_size,
        dropout=args.dropout,
        base_channels=args.base_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} (base_channels={args.base_channels})")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # lower is better for val_loss, higher is better for val_f1/val_acc
    best_metric = float("inf") if args.select_by == "val_loss" else float("-inf")
    history = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_correct = 0
        n_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total += x.size(0)

        train_loss = running_loss / max(1, n_total)
        train_acc = n_correct / max(1, n_total)

        val_loss, val_metrics = evaluate(model, val_loader, device, criterion)

        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_metrics.accuracy:.4f} "
            f"val_f1 {val_metrics.f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_metrics.accuracy,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall,
                "val_f1": val_metrics.f1,
            }
        )

        current_metric = {
            "val_loss": val_loss,
            "val_f1": val_metrics.f1,
            "val_acc": val_metrics.accuracy,
        }[args.select_by]
        is_better = (
            current_metric < best_metric
            if args.select_by == "val_loss"
            else current_metric > best_metric
        )
        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_metrics.accuracy,
            "val_f1": val_metrics.f1,
            "image_size": args.image_size,
            "dropout": args.dropout,
            "base_channels": args.base_channels,
            "select_by": args.select_by,
        }
        if is_better:
            best_metric = current_metric
            torch.save(ckpt_payload, ckpt_dir / "best_model.pt")
            print(f"  -> new best checkpoint saved ({args.select_by}={current_metric:.4f})")

        # always keep the latest checkpoint too
        torch.save(ckpt_payload, ckpt_dir / "last_model.pt")

    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.1f}s ({elapsed/60:.2f} min)")

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "elapsed_seconds": elapsed,
                "args": vars(args),
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "test_size": len(test_ds),
                "device": str(device),
            },
            f,
            indent=2,
        )
    print(f"Training log written to {log_path}")


if __name__ == "__main__":
    main()
