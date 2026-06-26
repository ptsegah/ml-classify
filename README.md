# Cat vs Dog Classifier

A binary image classifier that distinguishes cats from dogs using transfer learning with ResNet18 (PyTorch). Fine-tuning a pretrained model gives ~90–93% validation accuracy without needing a GPU for long.

## Project Structure

```
ml-classify/
├── app/
│   ├── config.py       # hyperparameters and paths
│   ├── dataset.py      # data loading and augmentation
│   ├── model.py        # ResNet18 with custom head
│   ├── train.py        # training loop
│   ├── predict.py      # inference
│   ├── main.py         # CLI entry point
│   └── requirements.txt
├── scripts/
│   └── prepare_data.py # optional: splits raw Kaggle data into train/val
├── data/               # images go here (auto-downloaded, not committed)
└── checkpoints/        # saved model weights (not committed)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
```

## Dataset

The [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (~800MB, ~7,400 images) is used. **No account or manual download needed** — it downloads automatically when you first run training.

> If you prefer the larger Kaggle Dogs vs Cats dataset (25k images), see `scripts/prepare_data.py` for setup instructions.

## Training

```bash
python app/main.py train --data-dir data
```

On first run the dataset downloads to `data/` automatically. Training then prints a per-epoch table and saves the best checkpoint to `checkpoints/best_model.pth`.

Optional flags:
- `--epochs 15` — override default (10)

## Inference

Single image:
```bash
python app/main.py predict --image path/to/photo.jpg
```

Batch (folder of images):
```bash
python app/main.py predict --folder path/to/images/
```

Example output:
```
cat.4521.jpg: cat (98.7%)
dog.1032.jpg: dog (96.2%)
```

## Model

- **Architecture**: ResNet18 pretrained on ImageNet, final layer replaced with `Linear(512, 2)`
- **Training strategy**: Feature extraction (backbone frozen, only head trained)
- **Optimizer**: Adam, lr=1e-4
- **Input**: 224×224 RGB, normalized with ImageNet mean/std
- **Dataset**: Oxford-IIIT Pet (~3,680 train / ~3,669 val)
- **Expected val accuracy**: ~90–93% after 10 epochs
