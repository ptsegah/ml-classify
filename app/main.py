import argparse
import sys
from pathlib import Path


def cmd_train(args):
    from train import train
    train(data_dir=args.data_dir, epochs=args.epochs)


def cmd_predict(args):
    from predict import load_model, predict_image, predict_folder

    if not args.image and not args.folder:
        print("Error: provide --image or --folder", file=sys.stderr)
        sys.exit(1)

    model, class_names = load_model(args.checkpoint)

    if args.image:
        label, confidence = predict_image(args.image, model, class_names)
        print(f"{Path(args.image).name}: {label} ({confidence:.1%})")

    if args.folder:
        predict_folder(args.folder, model, class_names)


def main():
    parser = argparse.ArgumentParser(description="Cat vs Dog Classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument("--data-dir", default="../data", help="Path to data/ directory")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.set_defaults(func=cmd_train)

    predict_p = sub.add_parser("predict", help="Run inference")
    predict_p.add_argument("--image", help="Path to a single image")
    predict_p.add_argument("--folder", help="Path to a folder of images")
    predict_p.add_argument("--checkpoint", default="../checkpoints/best_model.pth")
    predict_p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
