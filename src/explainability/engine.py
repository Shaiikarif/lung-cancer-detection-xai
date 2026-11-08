"""Unified Explainability Engine for multi-method interpretation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.explainability.gradcam import GradCAMExplainer
from src.explainability.lime_explainer import LIMExplainer
from src.explainability.integrated_gradients import IntegratedGradientsExplainer
from src.preprocessing.transforms import DEFAULT_MEAN, DEFAULT_STD
from src.utils.logger import get_logger

logger = get_logger("xai_engine")


class ExplainabilityEngine:
    """Unified engine coordinating multiple Explainable AI (XAI) algorithms."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
        mean: List[float] = DEFAULT_MEAN,
        std: List[float] = DEFAULT_STD
    ) -> None:
        """Initialize ExplainabilityEngine.

        Args:
            model: PyTorch model.
            device: Compute device.
            class_names: List of class labels (e.g. ['Normal', 'Benign', 'Malignant']).
            mean: Normalization channel means.
            std: Normalization channel standard deviations.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.class_names = class_names or ["Normal", "Benign", "Malignant"]
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.gradcam = GradCAMExplainer(model, device=self.device)
        self.lime = LIMExplainer(model, device=self.device, mean=mean, std=std)
        self.integrated_gradients = IntegratedGradientsExplainer(model, device=self.device)

    def load_image(
        self,
        image_path: Union[str, Path],
        img_size: int = 224
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Load and prepare image for explanation.

        Returns:
            rgb_uint8: (H, W, 3) uint8 image.
            rgb_float: (H, W, 3) float image in [0.0, 1.0].
            tensor: (1, 3, H, W) normalized PyTorch tensor.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image at path: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))

        rgb_float = img_resized.astype(np.float32) / 255.0
        normalized = (rgb_float - self.mean) / self.std

        tensor = torch.from_numpy(
            normalized.transpose(2, 0, 1)
        ).float().unsqueeze(0).to(self.device)

        return img_resized, rgb_float, tensor

    def explain(
        self,
        image_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None,
        include_lime: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive visual explanation panel.

        Args:
            image_path: Path to input CT scan.
            save_path: Optional path to save visual explanation panel plot.
            include_lime: Whether to compute LIME (computationally heavier).

        Returns:
            Dict containing predictions, Grad-CAM, Integrated Gradients, and optional LIME attributions.
        """
        rgb_uint8, rgb_float, img_tensor = self.load_image(image_path)

        # 1. Grad-CAM
        gcam_result = self.gradcam.explain(img_tensor, rgb_float, use_plus_plus=False)
        pred_class = gcam_result["predicted_class"]

        # 2. Grad-CAM++
        gcampp_result = self.gradcam.explain(img_tensor, rgb_float, target_class=pred_class, use_plus_plus=True)

        # 3. Integrated Gradients
        ig_result = self.integrated_gradients.explain(img_tensor, target_class=pred_class)

        # 4. Attention Weights from Model if available
        spatial_attention = None
        if hasattr(self.model, "attention") and hasattr(self.model.attention, "last_spatial_weights"):
            weights = self.model.attention.last_spatial_weights
            if weights is not None:
                spatial_attention = weights.squeeze().cpu().numpy()

        results = {
            "prediction": self.class_names[pred_class],
            "confidence": gcam_result["confidence"],
            "probabilities": gcam_result["probabilities"],
            "gradcam": gcam_result,
            "gradcam_plus_plus": gcampp_result,
            "integrated_gradients": ig_result,
            "spatial_attention": spatial_attention,
            "original_image": rgb_uint8
        }

        if include_lime:
            results["lime"] = self.lime.explain(rgb_uint8, target_class=pred_class, num_samples=300)

        if save_path:
            self._save_summary_plot(results, Path(save_path))

        return results

    def _save_summary_plot(self, results: Dict[str, Any], save_path: Path) -> None:
        """Plot and save multi-panel explanation figure."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(results["original_image"])
        axes[0].set_title(f"Input CT Scan\nPred: {results['prediction']} ({results['confidence']:.1%})", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(results["gradcam"]["overlay"])
        axes[1].set_title("Grad-CAM Overlay", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(results["gradcam_plus_plus"]["overlay"])
        axes[2].set_title("Grad-CAM++ Overlay", fontsize=12)
        axes[2].axis("off")

        axes[3].imshow(results["integrated_gradients"]["attribution_map"], cmap="hot")
        axes[3].set_title("Integrated Gradients", fontsize=12)
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved explanation panel to %s", save_path)
