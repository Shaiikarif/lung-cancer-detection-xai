"""Integrated Gradients attribution module."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger("integrated_gradients")


class IntegratedGradientsExplainer:
    """Computes Integrated Gradients attribution maps via linear path interpolation."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize IntegratedGradientsExplainer.

        Args:
            model: PyTorch model.
            device: Compute device.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[np.ndarray, int, float]]:
        """Compute pixel attribution map using Path-Integrated Gradients.

        Args:
            image_tensor: Normalized image tensor of shape (1, 3, H, W).
            target_class: Class index for which attributions are calculated.
            steps: Number of interpolation steps in Riemann sum.
            baseline: Baseline reference tensor. Defaults to all zeros.

        Returns:
            Dict containing normalized 2D attribution heatmap and target class.
        """
        image_tensor = image_tensor.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(image_tensor).to(self.device)
        else:
            baseline = baseline.to(self.device)

        if target_class is None:
            with torch.no_grad():
                logits = self.model(image_tensor)
                target_class = int(torch.argmax(logits, dim=1).item())

        alphas = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        accumulated_grads = torch.zeros_like(image_tensor)

        for alpha in alphas:
            interpolated = baseline + alpha * (image_tensor - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            output = self.model(interpolated)
            target_score = output[0, target_class]

            self.model.zero_grad()
            target_score.backward()

            if interpolated.grad is not None:
                accumulated_grads += interpolated.grad.detach()

        avg_grads = accumulated_grads / (steps + 1)
        integrated_grads = (image_tensor - baseline) * avg_grads
        integrated_grads_np = integrated_grads.squeeze(0).cpu().numpy()

        # Aggregate across color channels
        attribution = np.mean(np.abs(integrated_grads_np), axis=0)

        # Normalize to [0, 1] range
        attr_min = attribution.min()
        attr_max = attribution.max()
        if attr_max - attr_min > 1e-8:
            attribution = (attribution - attr_min) / (attr_max - attr_min)
        else:
            attribution = np.zeros_like(attribution)

        return {
            "attribution_map": attribution,
            "target_class": target_class
        }
