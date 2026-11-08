"""Adversarial robustness evaluation via FGSM (Fast Gradient Sign Method)."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.preprocessing.transforms import unnormalize_tensor, normalize_tensor, DEFAULT_MEAN, DEFAULT_STD
from src.utils.logger import get_logger

logger = get_logger("adversarial")


class AdversarialEvaluator:
    """Evaluates model vulnerability to gradient-based adversarial perturbations."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        mean: List[float] = DEFAULT_MEAN,
        std: List[float] = DEFAULT_STD
    ) -> None:
        """Initialize AdversarialEvaluator.

        Args:
            model: PyTorch model.
            device: Compute device.
            mean: Normalization mean values.
            std: Normalization standard deviation values.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.mean = mean
        self.std = std

    def fgsm_attack(
        self,
        normalized_image: torch.Tensor,
        target_label: int,
        epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate FGSM adversarial example with proper unnormalize -> perturb -> clamp -> normalize cycle.

        Args:
            normalized_image: Input tensor of shape (1, C, H, W) normalized with ImageNet stats.
            target_label: Ground truth integer class label.
            epsilon: Perturbation magnitude in [0.0, 1.0] pixel range.

        Returns:
            (adversarial_normalized_image, perturbation_tensor)
        """
        if epsilon == 0.0:
            return normalized_image.clone().detach(), torch.zeros_like(normalized_image)

        # Unnormalize to valid [0, 1] pixel space
        img_unnorm = unnormalize_tensor(normalized_image.to(self.device), self.mean, self.std)
        img_unnorm = img_unnorm.clone().detach().requires_grad_(True)

        # Re-normalize for forward pass through the model
        img_norm = normalize_tensor(img_unnorm, self.mean, self.std)

        self.model.eval()
        output = self.model(img_norm)
        label_tensor = torch.tensor([target_label], device=self.device)
        loss = F.cross_entropy(output, label_tensor)

        self.model.zero_grad()
        loss.backward()

        if img_unnorm.grad is None:
            logger.warning("Gradient was None during FGSM attack. Returning clean image.")
            return normalized_image.clone().detach(), torch.zeros_like(normalized_image)

        # Create perturbation in [0, 1] pixel space
        grad_sign = img_unnorm.grad.sign()
        perturbed_unnorm = img_unnorm + epsilon * grad_sign
        perturbed_unnorm = torch.clamp(perturbed_unnorm, 0.0, 1.0)

        # Renormalize for model evaluation
        adv_normalized = normalize_tensor(perturbed_unnorm.detach(), self.mean, self.std)
        perturbation = (perturbed_unnorm - img_unnorm).detach()

        return adv_normalized, perturbation

    def evaluate_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        epsilons: List[float] = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    ) -> Dict[float, float]:
        """Test accuracy across various perturbation strength levels.

        Args:
            dataloader: DataLoader providing (images, labels, paths).
            epsilons: List of perturbation strengths to evaluate.

        Returns:
            Dict mapping epsilon to classification accuracy.
        """
        self.model.eval()
        results: Dict[float, float] = {}

        logger.info("Evaluating adversarial robustness for epsilons: %s", epsilons)

        for eps in epsilons:
            correct = 0
            total = 0

            for images, labels, _ in tqdm(dataloader, desc=f"  ε={eps:.3f}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                for i in range(images.size(0)):
                    img = images[i:i + 1]
                    lbl = labels[i].item()

                    adv_img, _ = self.fgsm_attack(img, lbl, eps)

                    with torch.no_grad():
                        logits = self.model(adv_img)
                        pred = logits.argmax(dim=1).item()

                    if pred == lbl:
                        correct += 1
                    total += 1

            acc = correct / total if total > 0 else 0.0
            results[eps] = acc
            logger.info("Epsilon %.3f: Accuracy = %.4f (%d/%d)", eps, acc, correct, total)

        return results
