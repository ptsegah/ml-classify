import sys
from pathlib import Path
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from model import build_model


@pytest.fixture
def class_names():
    return ["cat", "dog"]


@pytest.fixture
def dummy_model():
    return build_model(num_classes=2)


@pytest.fixture
def dummy_image_path(tmp_path):
    img = Image.new("RGB", (64, 64), color=(120, 80, 60))
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return path


@pytest.fixture
def dummy_checkpoint(tmp_path, dummy_model):
    path = tmp_path / "best_model.pth"
    torch.save(
        {"epoch": 1, "model_state": dummy_model.state_dict(), "class_names": ["cat", "dog"]},
        path,
    )
    return path
