import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from .dataset import FashionProductDataset
from .models import get_model
from .utils import split_train_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/styles.csv")
    parser.add_argument("--img_dir", default="data/images")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("meta", {})
    model_name = meta.get("model_name", "mobilenet_v2")

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = FashionProductDataset(
        args.csv_path,
        args.img_dir,
        label_column="subCategory",
        top_k_classes=5,
        max_per_class=1000,
        transform=test_tf,
    )

    labels = dataset.labels
    _, val_idx = split_train_val(labels)
    test_ds = Subset(dataset, val_idx)

    test_loader = DataLoader(test_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name, num_classes=len(dataset.label_to_idx))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(device)
            labs = labs.to(device)

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = (all_preds == all_labels).mean()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {model_name}")
    print(f"Num test samples: {len(all_labels)}")
    print(f"Test accuracy: {acc:.4f}")

    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        digits=3,
    ))


if __name__ == "__main__":
    main()
