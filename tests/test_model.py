"""Tests for the CNN architecture: forward pass shape correctness."""
import torch

from src.model import EmergencyVehicleCNN


def test_forward_pass_output_shape_default():
    model = EmergencyVehicleCNN(num_classes=2, image_size=128)
    x = torch.randn(4, 3, 128, 128)
    out = model(x)
    assert out.shape == (4, 2)


def test_forward_pass_output_shape_small_resolution():
    model = EmergencyVehicleCNN(num_classes=2, image_size=64)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 2)


def test_forward_pass_single_sample():
    model = EmergencyVehicleCNN(num_classes=2, image_size=128)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 2)


def test_features_output_channels():
    model = EmergencyVehicleCNN(num_classes=2, image_size=128)
    x = torch.randn(2, 3, 128, 128)
    feats = model.features(x)
    # 4 conv blocks each halve spatial dims: 128 -> 8, final channel depth 256
    assert feats.shape == (2, 256, 8, 8)


def test_invalid_image_size_raises():
    import pytest

    with pytest.raises(ValueError):
        EmergencyVehicleCNN(num_classes=2, image_size=8)


def test_model_has_four_conv_layers():
    model = EmergencyVehicleCNN(num_classes=2, image_size=128)
    conv_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    assert len(conv_layers) == 4
    for layer in conv_layers:
        # each block: Conv2d, BatchNorm2d, ReLU, MaxPool2d
        assert isinstance(layer[0], torch.nn.Conv2d)
        assert isinstance(layer[1], torch.nn.BatchNorm2d)


def test_gradients_flow():
    model = EmergencyVehicleCNN(num_classes=2, image_size=128)
    x = torch.randn(2, 3, 128, 128)
    y = torch.tensor([0, 1])
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0
