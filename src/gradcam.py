import torch
import torch.nn.functional as F


class GradCAM:
    """
    Simple Grad-CAM implementation.

    Usage:
        cam = GradCAM(model, "features.18")      # for MobileNetV2
        cam = GradCAM(model, "layer4")           # for ResNet18
        heatmap = cam.generate(input_tensor)     # input_tensor: (1, C, H, W)
    """

    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None

        target_layer = self._get_target_layer(model, target_layer_name)

        def fwd_hook(module, inp, out):
            self.activations = out

        def bwd_hook(module, grad_input, grad_output):
            # grad_output is a tuple; we want the gradient wrt activations
            self.gradients = grad_output[0]

        # forward hook: grab activations
        target_layer.register_forward_hook(fwd_hook)
        # backward hook: grab gradients
        # use full backward hook to avoid future deprecation issues
        target_layer.register_full_backward_hook(bwd_hook)

    def _get_target_layer(self, model, name: str):
        """
        Resolve a dotted layer name like "features.18" or "layer4"
        to the actual PyTorch module.
        """
        current = model
        for part in name.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current

    def generate(self, input_tensor, target_class=None):
        """
        Generate a Grad-CAM heatmap for a single input.

        input_tensor: (1, C, H, W) torch tensor
        target_class: int or None. If None, uses argmax of logits.

        Returns:
            cam: (H_cam, W_cam) numpy array in [0, 1]
        """
        self.model.zero_grad()

        logits = self.model(input_tensor)  # (1, num_classes)

        if target_class is None:
            target_class = logits.argmax(dim=1)

        # assume batch size 1
        loss = logits[0, target_class]
        loss.backward()

        # activations: (1, C, H_cam, W_cam)
        # gradients:   (1, C, H_cam, W_cam)
        acts = self.activations[0]    # (C, H_cam, W_cam)
        grads = self.gradients[0]     # (C, H_cam, W_cam)

        # Global average pool gradients over spatial dims
        weights = grads.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activations
        cam = (weights[:, None, None] * acts).sum(dim=0)  # (H_cam, W_cam)
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam * 0.0
        return cam
