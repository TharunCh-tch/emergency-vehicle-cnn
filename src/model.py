"""
A 4-layer convolutional neural network, built from scratch in PyTorch, for
binary classification of emergency vs. non-emergency vehicles.

This mirrors the spirit of the original coursework architecture described on
LinkedIn: four conv blocks (conv -> batch norm -> ReLU -> max pool), dropout
for regularization, and a small fully-connected classifier head. It is not a
pretrained backbone -- every weight is trained from scratch on this project's
dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EmergencyVehicleCNN(nn.Module):
    """4-layer CNN with batch norm, dropout, and an FC classifier head.

    Input: (N, 3, image_size, image_size)
    Output: (N, num_classes) raw logits (use softmax/argmax or, for
        num_classes=1, sigmoid for inference).
    """

    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 128,
        dropout: float = 0.4,
        in_channels: int = 3,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.image_size = image_size

        def conv_block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        # 4 conv layers, channel depth doubling each block. base_channels
        # defaults to 16 (not the more typical 32) because this project's
        # dataset is small (~300 images) -- a 32-base model (4.6M params)
        # memorized the training set almost immediately (see results/results.md
        # run history); a narrower network generalizes better on this little
        # data without changing the 4-conv-block architecture shape.
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.layer1 = conv_block(in_channels, c1)
        self.layer2 = conv_block(c1, c2)
        self.layer3 = conv_block(c2, c3)
        self.layer4 = conv_block(c3, c4)

        # after 4 stride-2 pools, spatial size is image_size / 16
        reduced = image_size // 16
        if reduced < 1:
            raise ValueError("image_size too small for 4 pooling layers")
        flat_dim = c4 * reduced * reduced

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # keep a handle on the last conv layer for Grad-CAM
        self.gradcam_target_layer = self.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the activations of the final conv block (for Grad-CAM)."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == "__main__":
    model = EmergencyVehicleCNN(num_classes=2, image_size=128)
    dummy = torch.randn(4, 3, 128, 128)
    out = model(dummy)
    print("output shape:", out.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {n_params:,}")
