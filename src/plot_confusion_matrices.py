import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

classes = ["Bags", "Bottomwear", "Shoes", "Topwear", "Watches"]

cm_mobilenet = np.array([
    [200,   0,   0,   0,   0],
    [  1, 198,   0,   1,   0],
    [  0,   0, 200,   0,   0],
    [  0,   0,   0, 200,   0],
    [  0,   0,   0,   1, 199],
], dtype=float)

cm_resnet = np.array([
    [199,   0,   0,   0,   1],
    [  1, 196,   0,   3,   0],
    [  0,   0, 200,   0,   0],
    [  0,   0,   0, 200,   0],
    [  0,   0,   0,   0, 200],
], dtype=float)

def plot_cm(cm, title, filename):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)

plot_cm(cm_mobilenet, "Confusion Matrix – MobileNetV2",
        "confusion_mobilenetv2.png")

plot_cm(cm_resnet, "Confusion Matrix – ResNet18",
        "confusion_resnet18.png")
