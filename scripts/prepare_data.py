"""
Splits the raw Kaggle Dogs vs Cats dataset into train/val folders.

Usage:
    python scripts/prepare_data.py --raw-dir data/raw/train --out-dir data --val-split 0.2

Expected input structure:
    data/raw/train/
        cat.0.jpg
        cat.1.jpg
        ...
        dog.0.jpg
        dog.1.jpg
        ...

Output structure:
    data/
        train/
            cats/
            dogs/
        val/
            cats/
            dogs/
"""

import argparse
import random
import shutil
from pathlib import Path


def prepare(raw_dir: Path, out_dir: Path, val_split: float, seed: int = 42):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    all_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.jpeg"))
    if not all_files:
        raise FileNotFoundError(f"No .jpg files found in {raw_dir}")

    cats = sorted(p for p in all_files if p.name.startswith("cat"))
    dogs = sorted(p for p in all_files if p.name.startswith("dog"))
    print(f"Found {len(cats)} cat images and {len(dogs)} dog images")

    random.seed(seed)

    for label, files in [("cats", cats), ("dogs", dogs)]:
        random.shuffle(files)
        split = int(len(files) * val_split)
        val_files, train_files = files[:split], files[split:]

        for subset, subset_files in [("train", train_files), ("val", val_files)]:
            dest = out_dir / subset / label
            dest.mkdir(parents=True, exist_ok=True)
            for src in subset_files:
                shutil.copy2(src, dest / src.name)

        print(f"  {label}: {len(train_files)} train / {split} val")

    print(f"\nDone. Data written to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Prepare Dogs vs Cats dataset")
    parser.add_argument("--raw-dir", default="data/raw/train", help="Folder with raw Kaggle images")
    parser.add_argument("--out-dir", default="data", help="Output root directory")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    prepare(Path(args.raw_dir), Path(args.out_dir), args.val_split)


if __name__ == "__main__":
    main()
