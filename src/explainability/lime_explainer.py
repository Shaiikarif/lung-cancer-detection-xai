"""LIME (Local Interpretable Model-agnostic Explanations) module."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries

from src.preprocessing.transforms import DEFAULT_MEAN, DEFAULT_STD
from src.utils.logger import get_logger

logger = get_logger("lime")


class LIMExplainer:
    """Wrapper for generating LIME superpixel explanations for image classification."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        mean: List[float] = DEFAULT_MEAN,
        std: List[float] = DEFAULT_STD
    ) -> None:
        """Initialize LIMExplainer.

        Args:
            model: PyTorch model.
            device: Compute device.
            mean: Normalization channel means.
            std: Normalization channel standard deviations.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.explainer = lime_image.LimeImageExplainer()

    def _batch_predict(self, images: np.ndarray) -> np.ndarray:
        """Prediction callback function required by LIME.

        Args:
            images: Batch of RGB uint8 images with shape (N, H, W, 3).

        Returns:
            Probabilities matrix of shape (N, num_classes).
        """
        batch_tensors = []
        for img in images:
            img_float = img.astype(np.float32) / 255.0
            norm = (img_float - self.mean) / self.std
            t = torch.from_numpy(norm.transpose(2, 0, 1)).float()
            batch_tensors.append(t)

        batch_tensor = torch.stack(batch_tensors).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def explain(
        self,
        rgb_image_uint8: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: int = 500,
        num_features: int = 10
    ) -> Dict[str, Union[np.ndarray, int, float]]:
        """Generate LIME explanation for an image.

        Args:
            rgb_image_uint8: RGB image as uint8 array (H, W, 3).
            target_class: Target class index. If None, top predicted class is used.
            num_samples: Number of perturbation samples.
            num_features: Maximum number of superpixel features to highlight.

        Returns:
            Dict containing positive feature boundary image and full explanation mask.
        """
        explanation = self.explainer.explain_instance(
            rgb_image_uint8,
            self._batch_predict,
            top_labels=3,
            hide_color=0,
            num_samples=num_samples
        )

        if target_class is None:
            target_class = int(explanation.top_labels[0])

        temp_pos, mask_pos = explanation.get_image_and_mask(
            target_class,
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        lime_positive = mark_boundaries(temp_pos / 255.0, mask_pos, color=(0, 1, 0))

        temp_all, mask_all = explanation.get_image_and_mask(
            target_class,
            positive_only=False,
            num_features=num_features,
            hide_rest=False
        )
        lime_all = mark_boundaries(temp_all / 255.0, mask_all, color=(1, 0, 0))

        return {
            "target_class": target_class,
            "lime_positive": lime_positive,
            "lime_all": lime_all,
            "mask_positive": mask_pos,
            "mask_all": mask_all
        }
