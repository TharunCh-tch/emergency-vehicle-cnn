"""
CLI inference script: load a trained checkpoint and classify a new image as
emergency / non_emergency.

Usage:
    python -m src.infer --checkpoint checkpoints/best_model.pt --image path/to/photo.jpg
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image

from src.dataset import CLASS_NAMES, build_transforms
from src.model import EmergencyVehicleCNN


def load_model(checkpoint_path: str, device: torch.device) -> tuple[EmergencyVehicleCNN, int]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    image_size = ckpt.get("image_size", 128)
    model = EmergencyVehicleCNN(
        num_classes=2,
        image_size=image_size,
        dropout=ckpt.get("dropout", 0.4),
        base_channels=ckpt.get("base_channels", 16),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, image_size


def predict(model: EmergencyVehicleCNN, image_path: str, image_size: int, device: torch.device):
    transform = build_transforms(image_size, train=False)
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
    return CLASS_NAMES[pred_idx], float(probs[pred_idx].item()), probs.tolist()


def main():
    ap = argparse.ArgumentParser(description="Classify an image as emergency / non_emergency.")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    ap.add_argument("--image", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cpu")
    model, image_size = load_model(args.checkpoint, device)
    label, confidence, all_probs = predict(model, args.image, image_size, device)

    print(f"Image: {args.image}")
    print(f"Prediction: {label}  (confidence: {confidence:.4f})")
    for name, p in zip(CLASS_NAMES, all_probs):
        print(f"  {name}: {p:.4f}")


if __name__ == "__main__":
    main()
