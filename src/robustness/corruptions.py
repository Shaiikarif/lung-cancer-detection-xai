"""Image corruption and domain shift robustness evaluation."""

from typing import Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.preprocessing.transforms import unnormalize_tensor, normalize_tensor, DEFAULT_MEAN, DEFAULT_STD
from src.utils.logger import get_logger

logger = get_logger("corruptions")


class CorruptionEvaluator:
    """Tests model resilience against common imaging corruptions and artifacts."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        mean: List[float] = DEFAULT_MEAN,
        std: List[float] = DEFAULT_STD
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.mean = mean
        self.std = std

    @staticmethod
    def _add_noise(x_unnorm: torch.Tensor, std: float) -> torch.Tensor:
        """Add zero-mean Gaussian noise in [0, 1] space."""
        noise = torch.randn_like(x_unnorm) * std
        return torch.clamp(x_unnorm + noise, 0.0, 1.0)

    @staticmethod
    def _blur(x_unnorm: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply 2D average spatial blur."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        return pool(x_unnorm)

    @staticmethod
    def _occlude(x_unnorm: torch.Tensor, ratio: float) -> torch.Tensor:
        """Simulate local radio-opaque occlusion artifact."""
        result = x_unnorm.clone()
        _, _, h, w = x_unnorm.shape
        mask_h = max(1, int(h * ratio))
        mask_w = max(1, int(w * ratio))
        top = np.random.randint(0, max(1, h - mask_h))
        left = np.random.randint(0, max(1, w - mask_w))
        result[:, :, top:top + mask_h, left:left + mask_w] = 0.0
        return result

    def get_corruptions(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        """Return registry of corruption transformations operating in [0, 1] pixel space."""
        return {
            "Clean (Baseline)": lambda x: x,
            "Gaussian Noise (σ=0.05)": lambda x: self._add_noise(x, 0.05),
            "Gaussian Noise (σ=0.10)": lambda x: self._add_noise(x, 0.10),
            "Gaussian Blur (k=3)": lambda x: self._blur(x, 3),
            "Gaussian Blur (k=5)": lambda x: self._blur(x, 5),
            "Brightness Shift (+20%)": lambda x: torch.clamp(x * 1.2, 0.0, 1.0),
            "Brightness Shift (-20%)": lambda x: torch.clamp(x * 0.8, 0.0, 1.0),
            "Spatial Occlusion (10%)": lambda x: self._occlude(x, 0.10),
            "Spatial Occlusion (20%)": lambda x: self._occlude(x, 0.20),
        }

    def evaluate_dataset(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate accuracy under diverse synthetic corruptions.

        Args:
            dataloader: DataLoader with normalized images.

        Returns:
            Dict mapping corruption name to accuracy.
        """
        self.model.eval()
        corruptions = self.get_corruptions()
        results: Dict[str, float] = {}

        for name, corruption_fn in corruptions.items():
            correct = 0
            total = 0

            for images, labels, _ in tqdm(dataloader, desc=f"  {name}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Unnormalize to [0, 1] before applying image corruption
                unnorm = unnormalize_tensor(images, self.mean, self.std)
                corrupted_unnorm = corruption_fn(unnorm)
                # Renormalize for model input
                corrupted_norm = normalize_tensor(corrupted_unnorm, self.mean, self.std)

                with torch.no_grad():
                    outputs = self.model(corrupted_norm)
                    preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = correct / total if total > 0 else 0.0
            results[name] = acc
            logger.info("%-30s: Accuracy = %.4f", name, acc)

        return results
