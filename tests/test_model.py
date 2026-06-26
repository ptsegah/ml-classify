import torch
import torch.nn as nn
from model import build_model, unfreeze_model


def test_build_model_returns_module():
    model = build_model()
    assert isinstance(model, nn.Module)


def test_output_shape_default():
    model = build_model(num_classes=2)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2)


def test_output_shape_custom_classes():
    model = build_model(num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5)


def test_backbone_frozen_after_build():
    model = build_model()
    for name, param in model.named_parameters():
        if "fc" not in name:
            assert not param.requires_grad, f"{name} should be frozen"


def test_fc_trainable_after_build():
    model = build_model()
    for param in model.fc.parameters():
        assert param.requires_grad


def test_unfreeze_makes_all_trainable():
    model = build_model()
    unfreeze_model(model)
    for name, param in model.named_parameters():
        assert param.requires_grad, f"{name} should be trainable after unfreeze"
