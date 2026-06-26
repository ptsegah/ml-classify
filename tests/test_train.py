import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train import _evaluate


def _fake_loader(num_batches=2, batch_size=4, num_classes=2):
    images = torch.randn(num_batches * batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_batches * batch_size,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_evaluate_returns_float_loss(dummy_model):
    loader = _fake_loader()
    criterion = nn.CrossEntropyLoss()
    loss, _ = _evaluate(dummy_model, loader, criterion)
    assert isinstance(loss, float)


def test_evaluate_returns_int_correct(dummy_model):
    loader = _fake_loader()
    criterion = nn.CrossEntropyLoss()
    _, correct = _evaluate(dummy_model, loader, criterion)
    assert isinstance(correct, int)


def test_evaluate_correct_within_bounds(dummy_model):
    loader = _fake_loader(num_batches=2, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    _, correct = _evaluate(dummy_model, loader, criterion)
    assert 0 <= correct <= 8


def test_evaluate_loss_is_positive(dummy_model):
    loader = _fake_loader()
    criterion = nn.CrossEntropyLoss()
    loss, _ = _evaluate(dummy_model, loader, criterion)
    assert loss > 0
