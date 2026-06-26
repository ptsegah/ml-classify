import torch
from PIL import Image
from torchvision import transforms
from dataset import CLASS_NAMES
import config


def test_class_names():
    assert CLASS_NAMES == ["cat", "dog"]


def test_class_names_length():
    assert len(CLASS_NAMES) == 2


def _make_pil_image():
    return Image.new("RGB", (256, 256), color=(100, 150, 200))


def test_train_transform_output_shape():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    tensor = transform(_make_pil_image())
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_val_transform_output_shape():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    tensor = transform(_make_pil_image())
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_transform_output_is_tensor():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    tensor = transform(_make_pil_image())
    assert isinstance(tensor, torch.Tensor)
