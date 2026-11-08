"""ResNet50 model with CBAM Attention for Lung Cancer Classification."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.models.attention import AttentionModule
from src.utils.logger import get_logger

logger = get_logger("model")


class LungCancerModel(nn.Module):
    """Deep learning model for lung cancer classification.

    Architecture:
    1. Pretrained ResNet-50 backbone (feature extractor)
    2. CBAM Attention module (channel & spatial focus)
    3. Global Adaptive Average Pooling
    4. Custom MLP classification head (2048 -> 512 -> 128 -> num_classes)
       with BatchNorm1d, ReLU, and Dropout regularization.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        enable_cam_hook: bool = False
    ) -> None:
        """Initialize LungCancerModel.

        Args:
            num_classes: Number of target categories (Normal, Benign, Malignant).
            pretrained: Whether to load ImageNet pretrained weights for ResNet50.
            dropout_rate: Dropout rate for classifier regularisation.
            enable_cam_hook: Whether to enable direct backward gradient saving for Grad-CAM.
        """
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)
        self.feature_dim = self.backbone.fc.in_features  # 2048

        # Replace classification head
        self.backbone.fc = nn.Identity()

        # Attention module
        self.attention = AttentionModule(self.feature_dim)

        # Custom MLP classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # Storage for feature maps and gradients (Grad-CAM support)
        self.feature_maps: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._enable_cam_hook: bool = enable_cam_hook
        self._hook_handle = None

    def _save_gradient(self, grad: torch.Tensor) -> None:
        """Hook to capture activation gradients for Grad-CAM."""
        self.gradients = grad

    def set_cam_hook_enabled(self, enabled: bool) -> None:
        """Toggle Grad-CAM gradient tracking safely."""
        self._enable_cam_hook = enabled
        if not enabled and self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Class logits tensor of shape (batch_size, num_classes).
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Store feature maps before attention
        self.feature_maps = x

        # Register gradient hook only when explicitly enabled and tensor requires grad
        if self._enable_cam_hook and x.requires_grad:
            if self._hook_handle is not None:
                self._hook_handle.remove()
            self._hook_handle = x.register_hook(self._save_gradient)

        # Apply CBAM Attention
        x = self.attention(x)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        # Classifier
        output = self.classifier(x)
        return output


def create_model(
    num_classes: int = 3,
    pretrained: bool = True,
    dropout_rate: float = 0.5
) -> LungCancerModel:
    """Factory function to instantiate LungCancerModel with parameter summary.

    Args:
        num_classes: Output class count.
        pretrained: Whether to load ImageNet pretraining.
        dropout_rate: Dropout rate for classifier.

    Returns:
        Instantiated LungCancerModel.
    """
    model = LungCancerModel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model: ResNet50 + CBAM Attention + Custom Classifier")
    logger.info("Total Parameters: %s | Trainable: %s", f"{total_params:,}", f"{trainable_params:,}")
    return model
