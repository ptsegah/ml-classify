import torch
import torch.nn as nn
from pathlib import Path
import config
from dataset import get_dataloaders
from model import build_model


def train(data_dir: str | Path = config.DATA_DIR, epochs: int = config.EPOCHS):
    data_dir = Path(data_dir)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = get_dataloaders(data_dir)
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_loader.dataset)} images  |  Val: {len(val_loader.dataset)} images")
    print(f"Device: {config.DEVICE}\n")

    model = build_model(num_classes=len(class_names)).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=config.LR)

    best_val_acc = 0.0

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        val_loss, val_correct = _evaluate(model, val_loader, criterion)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.1%}  {val_loss:>8.4f}  {val_acc:>6.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "class_names": class_names},
                config.CHECKPOINT_PATH,
            )
            print(f"         -> Saved best model (val acc {val_acc:.1%})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1%}")
    print(f"Checkpoint: {config.CHECKPOINT_PATH}")
    return model, class_names


def _evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    total_loss /= len(loader.dataset)
    return total_loss, correct
