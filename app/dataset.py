from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config

CLASS_NAMES = ["cat", "dog"]


def get_dataloaders(data_dir=config.DATA_DIR):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    train_dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="binary-category",
        transform=train_transform,
        download=True,
    )
    val_dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="binary-category",
        transform=val_transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, CLASS_NAMES
