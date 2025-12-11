from torchvision import models
import torch.nn as nn


def get_model(model_name, num_classes, pretrained=True):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
