"""
Grad-CAM (Gradient-weighted Class Activation Mapping), implemented from
scratch with PyTorch forward/backward hooks.

Reference (method, not code): Selvaraju et al., "Grad-CAM: Visual
Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
https://arxiv.org/abs/1610.02391

Usage (produces saved heatmap overlays for a handful of test images):
    python -m src.gradcam --checkpoint checkpoints/best_model.pt --num-samples 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import CLASS_NAMES, get_splits
from src.model import EmergencyVehicleCNN


class GradCAM:
    """Grad-CAM for a single target conv layer of a CNN."""

    def __init__(self, model: EmergencyVehicleCNN, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """x: (1, C, H, W). Returns (heatmap in [0,1] at input resolution, predicted class)."""
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()

        activations = self.activations[0]  # (C, h, w)
        gradients = self.gradients[0]  # (C, h, w)
        weights = gradients.mean(dim=(1, 2))  # global-average-pool gradients -> (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for c, w in enumerate(weights):
            cam += w * activations[c]
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze().numpy(), class_idx

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_heatmap(orig_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a [0,1] heatmap on a PIL image using a simple red-hot colormap
    (no matplotlib dependency needed for the overlay itself)."""
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    # simple colormap: red channel = heatmap, green = heatmap/2, blue = inverted
    h, w = heatmap_u8.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[..., 0] = heatmap_u8  # R
    color[..., 1] = (heatmap_u8 * 0.4).astype(np.uint8)  # G
    color[..., 2] = 255 - heatmap_u8  # B
    color_img = Image.fromarray(color).resize(orig_img.size)
    blended = Image.blend(orig_img.convert("RGB"), color_img, alpha=alpha)
    return blended


def denormalize_to_pil(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/raw")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default="results/gradcam")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = ckpt.get("image_size", 128)
    model = EmergencyVehicleCNN(
        num_classes=2,
        image_size=image_size,
        dropout=ckpt.get("dropout", 0.4),
        base_channels=ckpt.get("base_channels", 16),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_ds = get_splits(Path(args.data_root), image_size=image_size, seed=args.seed)

    cam = GradCAM(model, model.gradcam_target_layer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num_samples, len(test_ds))
    for i in range(n):
        x, y = test_ds[i]
        x_batch = x.unsqueeze(0)
        heatmap, pred_class = cam(x_batch)
        orig = denormalize_to_pil(x)
        overlay = overlay_heatmap(orig, heatmap)

        true_name = CLASS_NAMES[y]
        pred_name = CLASS_NAMES[pred_class]
        fname = out_dir / f"gradcam_{i:02d}_true-{true_name}_pred-{pred_name}.jpg"

        # side-by-side: original | overlay
        combo = Image.new("RGB", (orig.width * 2 + 8, orig.height), color=(255, 255, 255))
        combo.paste(orig, (0, 0))
        combo.paste(overlay, (orig.width + 8, 0))
        combo.save(fname, quality=88)
        print(f"saved {fname}")

    cam.remove()
    print(f"Grad-CAM samples written to {out_dir}")


if __name__ == "__main__":
    main()
