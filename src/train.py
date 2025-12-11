import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .dataset import FashionProductDataset
from .models import get_model
from .utils import split_train_val, run_epoch, save_checkpoint


def main():
    print("Starting train.py main()")

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/styles.csv")
    parser.add_argument("--img_dir", default="data/images")
    parser.add_argument("--model_name", default="mobilenet_v2",
                        choices=["mobilenet_v2", "resnet18"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # -----------------------
    # Transforms
    # -----------------------
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(.1, .1, .1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, .456, .406], [.229, .224, .225])
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, .456, .406], [.229, .224, .225])
    ])

    # -----------------------
    # Load Dataset
    # -----------------------
    print("Loading dataset...")

    dataset = FashionProductDataset(
        args.csv_path,
        args.img_dir,
        label_column="subCategory",
        top_k_classes=5,
        max_per_class=1000,
        transform=None
    )

    print("Dataset loaded:", len(dataset), "items")

    labels = dataset.labels
    train_idx, val_idx = split_train_val(labels)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    dataset.transform = train_tf
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    dataset.transform = test_tf
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # -----------------------
    # Model setup
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = get_model(args.model_name, num_classes=len(dataset.label_to_idx))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    ckpt_path = f"checkpoints/{args.model_name}_best.pth"

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            train_loader, model, loss_fn, opt, device, training=True
        )

        val_loss, val_acc = run_epoch(
            val_loader, model, loss_fn, opt, device, training=False
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, ckpt_path, {
                "model_name": args.model_name,
                "label_to_idx": dataset.label_to_idx
            })
            print(f"Saved checkpoint at {ckpt_path} (best acc: {best_acc:.4f})")


if __name__ == "__main__":
    main()
