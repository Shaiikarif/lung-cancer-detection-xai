"""Monte Carlo Dropout Uncertainty Quantification (UQ)."""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger("uncertainty")


class UncertaintyQuantifier:
    """Estimates epistemic prediction uncertainty via Monte Carlo (MC) Dropout sampling."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
        uncertainty_threshold: float = 0.15,
        confidence_threshold: float = 0.70
    ) -> None:
        """Initialize UncertaintyQuantifier.

        Args:
            model: PyTorch model containing nn.Dropout layers.
            device: Compute device.
            class_names: Target class names.
            uncertainty_threshold: Maximum acceptable standard deviation before flagging as uncertain.
            confidence_threshold: Minimum acceptable mean probability before flagging as low confidence.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.class_names = class_names or ["Normal", "Benign", "Malignant"]
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _enable_dropout(model: torch.nn.Module) -> None:
        """Put all dropout layers into train mode to sample stochastic sub-networks."""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    def estimate(
        self,
        image_tensor: torch.Tensor,
        num_samples: int = 30
    ) -> Dict[str, Union[str, float, bool, np.ndarray]]:
        """Run T forward passes with active dropout to compute posterior predictive statistics.

        Args:
            image_tensor: Normalized tensor of shape (1, 3, H, W).
            num_samples: Number of stochastic passes (T).

        Returns:
            Dict containing mean probabilities, standard deviations, entropy, and reliability flag.
        """
        image_tensor = image_tensor.to(self.device)

        self.model.eval()
        self._enable_dropout(self.model)

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                logits = self.model(image_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy()[0])

        self.model.eval()

        predictions_arr = np.array(predictions)  # Shape (T, num_classes)
        mean_probs = predictions_arr.mean(axis=0)
        std_probs = predictions_arr.std(axis=0)

        pred_class_idx = int(np.argmax(mean_probs))
        confidence = float(mean_probs[pred_class_idx])
        uncertainty = float(std_probs[pred_class_idx])

        # Shannon Entropy: H(p) = -sum(p * log(p + eps))
        entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-10)))

        # A prediction is flagged as reliable only if confidence is high AND variance is low
        is_reliable = bool((uncertainty < self.uncertainty_threshold) and (confidence > self.confidence_threshold))

        return {
            "predicted_class": self.class_names[pred_class_idx],
            "predicted_class_idx": pred_class_idx,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "entropy": entropy,
            "is_reliable": is_reliable,
            "mean_probs": mean_probs,
            "std_probs": std_probs,
            "sample_predictions": predictions_arr
        }
