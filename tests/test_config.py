from pathlib import Path
import config


def test_image_size():
    assert config.IMAGE_SIZE == 224


def test_batch_size_positive():
    assert config.BATCH_SIZE > 0


def test_learning_rate_valid():
    assert 0 < config.LR < 1


def test_device_valid():
    assert config.DEVICE in ("cuda", "cpu")


def test_imagenet_stats_length():
    assert len(config.IMAGENET_MEAN) == 3
    assert len(config.IMAGENET_STD) == 3


def test_paths_are_path_objects():
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.CHECKPOINT_DIR, Path)
    assert isinstance(config.CHECKPOINT_PATH, Path)
