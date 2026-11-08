"""Grad-CAM and Grad-CAM++ visualization module."""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.preprocessing.transforms import DEFAULT_MEAN, DEFAULT_STD
from src.utils.logger import get_logger

logger = get_logger("gradcam")


class GradCAMExplainer:
    """Wrapper for generating Grad-CAM and Grad-CAM++ heatmaps on CNN models."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize GradCAMExplainer.

        Args:
            model: PyTorch model.
            target_layer: Layer to compute gradients against. Defaults to model.backbone.layer4[-1].
            device: Compute device (CPU or CUDA).
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

        if target_layer is None:
            # Default to last conv layer in ResNet50
            if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
                self.target_layer = model.backbone.layer4[-1]
            else:
                raise ValueError("Target layer must be specified if model does not have backbone.layer4")
        else:
            self.target_layer = target_layer

    def explain(
        self,
        image_tensor: torch.Tensor,
        rgb_image_float: np.ndarray,
        target_class: Optional[int] = None,
        use_plus_plus: bool = False
    ) -> Dict[str, Union[np.ndarray, int, float]]:
        """Generate Grad-CAM or Grad-CAM++ attribution heatmap.

        Args:
            image_tensor: Normalized tensor of shape (1, 3, H, W).
            rgb_image_float: RGB float image array of shape (H, W, 3) in [0.0, 1.0].
            target_class: Category index to explain. Defaults to highest probability class.
            use_plus_plus: If True, uses Grad-CAM++; otherwise uses standard Grad-CAM.

        Returns:
            Dict containing 'heatmap' (grayscale 2D), 'overlay' (RGB visualization),
            'target_class', and 'prediction_prob'.
        """
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = int(torch.argmax(probs).item())

        if target_class is None:
            target_class = pred_class

        cam_cls = GradCAMPlusPlus if use_plus_plus else GradCAM
        targets = [ClassifierOutputTarget(target_class)]

        with cam_cls(model=self.model, target_layers=[self.target_layer]) as cam:
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

        # Overlay on RGB display image
        overlay = show_cam_on_image(rgb_image_float, grayscale_cam, use_rgb=True)

        return {
            "heatmap": grayscale_cam,
            "overlay": overlay,
            "target_class": target_class,
            "predicted_class": pred_class,
            "confidence": float(probs[pred_class].item()),
            "probabilities": probs.cpu().numpy()
        }
