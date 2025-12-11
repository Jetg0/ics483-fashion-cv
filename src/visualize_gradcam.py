import argparse
import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

from .dataset import FashionProductDataset
from .models import get_model
from .gradcam import GradCAM


def overlay(img, cam):
    """
    Blend the original image and the Grad-CAM heatmap.

    img: torch tensor (C, H, W), normalized
    cam: numpy array (H_cam, W_cam) in [0, 1]
    """
    c, H, W = img.shape

    # Upsample cam to (H, W)
    cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # (1,1,H_cam,W_cam)
    cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
    cam_up = cam_up.squeeze().cpu().numpy()  # (H, W)

    # Convert image to [0,1] for display
    img_np = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    # Colorize heatmap
    heat = plt.cm.jet(cam_up)[..., :3]  # (H, W, 3)

    out = 0.4 * heat + 0.6 * img_np
    out = np.clip(out, 0, 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv_path", default="data/styles.csv")
    parser.add_argument("--img_dir", default="data/images")
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    # Load checkpoint metadata
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("meta", {})
    model_name = meta.get("model_name", "mobilenet_v2")
    label_to_idx = meta.get("label_to_idx")
    if label_to_idx is None:
        raise ValueError("Checkpoint meta is missing 'label_to_idx'.")
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Transform (same as eval)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, .456, .406], [.229, .224, .225])
    ])

    dataset = FashionProductDataset(
        args.csv_path,
        args.img_dir,
        label_column="subCategory",
        top_k_classes=len(label_to_idx),
        transform=transform,
    )

    # Rebuild model and load weights
    model = get_model(model_name, num_classes=len(label_to_idx))
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Pick correct target layer name
    # For torchvision models:
    #   - MobileNetV2: final conv features at model.features[18]
    #   - ResNet18: last conv block is model.layer4
    if model_name == "resnet18":
        target_layer_name = "layer4"
    else:
        # default to mobilenet_v2 style
        target_layer_name = "features.18"

    cam_gen = GradCAM(model, target_layer_name)

    # Make output dir
    os.makedirs("plots", exist_ok=True)

    # Random sample of dataset indices
    indices = random.sample(range(len(dataset)), args.samples)

    for i, idx in enumerate(indices):
        img, label = dataset[idx]       # img: (C,H,W), label: int
        inp = img.unsqueeze(0)          # batch size 1

        cam = cam_gen.generate(inp)     # (h_cam, w_cam) in [0,1]
        vis = overlay(img, cam)

        plt.figure(figsize=(4, 4))
        plt.imshow(vis)
        plt.title(f"True: {idx_to_label[int(label)]}")
        plt.axis("off")

        out_path = os.path.join("plots", f"gradcam_{model_name}_{i}.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print(f"Saved {len(indices)} Grad-CAM images to 'plots/' for {model_name}.")


if __name__ == "__main__":
    main()
