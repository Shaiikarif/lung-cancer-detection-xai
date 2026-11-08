"""Custom loss functions for class-imbalanced medical image classification."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance by down-weighting easy examples.

    Loss formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    References:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal loss for dense object detection. In ICCV 2017.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ) -> None:
        """Initialize FocalLoss.

        Args:
            alpha: Optional class weights tensor of shape (num_classes,).
            gamma: Focusing parameter balancing easy vs. hard examples.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss.

        Args:
            inputs: Logits tensor of shape (batch_size, num_classes).
            targets: Integer ground truth tensor of shape (batch_size,).

        Returns:
            Scalar focal loss tensor (if reduction is 'mean' or 'sum').
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # Probability of ground truth class
        focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(inputs.device)
            alpha_t = alpha_tensor[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
