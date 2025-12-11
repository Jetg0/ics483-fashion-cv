import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def split_train_val(labels, ratio=0.2):
    """
    Split indices into train/val with stratification.
    labels: 1D numpy array of class indices
    """
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=ratio,
        stratify=labels,
        random_state=42,
    )
    return train_idx, val_idx


def run_epoch(dataloader, model, loss_fn, opt, device, training=True):
    """
    One training or validation epoch.
    Returns (avg_loss, accuracy)
    """
    if training:
        model.train()
    else:
        model.eval()

    total = 0
    correct = 0
    running_loss = 0.0

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        if training:
            opt.zero_grad()

        with torch.set_grad_enabled(training):
            preds = model(imgs)
            loss = loss_fn(preds, labels)

            if training:
                loss.backward()
                opt.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = preds.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def save_checkpoint(model, path, metadata=None):
    """
    Save model weights plus optional metadata.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {"model": model.state_dict()}
    if metadata is not None:
        to_save["meta"] = metadata
    torch.save(to_save, path)
