from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import config
from model import build_model

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(checkpoint_path: str | Path = config.CHECKPOINT_PATH):
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names)).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names


def predict_image(image_path: str | Path, model, class_names: list[str]) -> tuple[str, float]:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    idx = probs.argmax().item()
    return class_names[idx], probs[idx].item()


def predict_folder(folder_path: str | Path, model, class_names: list[str]) -> None:
    folder = Path(folder_path)
    image_paths = [p for p in folder.iterdir() if p.suffix.lower() in _EXTENSIONS]

    if not image_paths:
        print(f"No images found in {folder}")
        return

    print(f"{'File':<40}  {'Prediction':<8}  Confidence")
    print("-" * 62)
    for path in sorted(image_paths):
        label, confidence = predict_image(path, model, class_names)
        print(f"{path.name:<40}  {label:<8}  {confidence:.1%}")
