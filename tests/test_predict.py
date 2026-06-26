import pytest
from predict import predict_image, predict_folder, load_model
from dataset import CLASS_NAMES


def test_predict_image_returns_valid_label(dummy_image_path, dummy_model):
    label, confidence = predict_image(dummy_image_path, dummy_model, CLASS_NAMES)
    assert label in CLASS_NAMES


def test_predict_image_confidence_in_range(dummy_image_path, dummy_model):
    _, confidence = predict_image(dummy_image_path, dummy_model, CLASS_NAMES)
    assert 0.0 <= confidence <= 1.0


def test_predict_folder_empty_dir(tmp_path, dummy_model, capsys):
    predict_folder(tmp_path, dummy_model, CLASS_NAMES)
    captured = capsys.readouterr()
    assert "No images found" in captured.out


def test_predict_folder_with_image(tmp_path, dummy_image_path, dummy_model):
    import shutil
    shutil.copy(dummy_image_path, tmp_path / "test.jpg")
    predict_folder(tmp_path, dummy_model, CLASS_NAMES)


def test_load_model_returns_model_and_classes(dummy_checkpoint):
    model, class_names = load_model(dummy_checkpoint)
    assert class_names == CLASS_NAMES
    assert hasattr(model, "forward")
