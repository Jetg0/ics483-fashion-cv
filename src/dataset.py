import os
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class FashionProductDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        label_column="subCategory",
        top_k_classes=5,
        max_per_class=None,
        transform=None,
    ):
        self.transform = transform
        self.img_dir = img_dir

        df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        df = df.dropna(subset=[label_column])

        df["id"] = df["id"].astype(str)
        df["img_path"] = df["id"].apply(lambda x: os.path.join(img_dir, x + ".jpg"))
        df = df[df["img_path"].apply(os.path.exists)]

        class_counts = df[label_column].value_counts()
        top_classes = class_counts.head(top_k_classes).index.tolist()
        df = df[df[label_column].isin(top_classes)]

        if max_per_class:
            df = (
                df.groupby(label_column, group_keys=False)
                .apply(
                    lambda x: x.sample(
                        min(len(x), max_per_class), random_state=42
                    )
                )
                .reset_index(drop=True)
            )

        labels = sorted(df[label_column].unique())
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        df["label_idx"] = df[label_column].map(self.label_to_idx)

        self.df = df
        self.labels = df["label_idx"].values.astype(np.int64)
        self.paths = df["img_path"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = int(self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label
