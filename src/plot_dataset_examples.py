import os
import random

import matplotlib.pyplot as plt
from torchvision import transforms

from .dataset import FashionProductDataset


def main():
    # Paths must match your training setup
    csv_path = "data/styles.csv"
    img_dir = "data/images"

    # Simple transform (no normalization) so colors look normal
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Same dataset config you used for training
    dataset = FashionProductDataset(
        csv_path,
        img_dir,
        label_column="subCategory",
        top_k_classes=5,
        max_per_class=1000,
        transform=transform,
    )

    # Build mapping: class_idx -> list of dataset indices
    by_class = {}
    for i, y in enumerate(dataset.labels):
        y = int(y)
        by_class.setdefault(y, []).append(i)

    # How many examples per class to show
    samples_per_class = 2
    class_indices = sorted(by_class.keys())
    num_classes = len(class_indices)

    # Make sure we have a plots folder
    os.makedirs("plots", exist_ok=True)

    # Create a grid: one row per class, 2 examples per row
    fig, axes = plt.subplots(
        num_classes, samples_per_class,
        figsize=(3 * samples_per_class, 3 * num_classes)
    )

    # If samples_per_class == 1, axes is 1D; normalize shape
    if samples_per_class == 1:
        axes = axes[:, None]

    # Map back from index -> class name (Bags, Bottomwear, etc.)
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}

    for row_idx, class_idx in enumerate(class_indices):
        idx_list = by_class[class_idx]
        k = min(samples_per_class, len(idx_list))
        chosen_indices = random.sample(idx_list, k=k)

        for col_idx, ds_idx in enumerate(chosen_indices):
            img, label = dataset[ds_idx]

            # img: C x H x W -> H x W x C, scale to [0,1]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            ax = axes[row_idx, col_idx]
            ax.imshow(img_np)
            ax.set_xticks([])
            ax.set_yticks([])

            # Put class name on left side of each row
            if col_idx == 0:
                class_name = idx_to_label[class_idx]
                ax.set_ylabel(class_name, fontsize=10)

    plt.tight_layout()
    out_path = os.path.join("plots", "fig_examples.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved example grid to {out_path}")


if __name__ == "__main__":
    main()
