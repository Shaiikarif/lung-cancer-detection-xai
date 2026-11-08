"""CBAM (Convolutional Block Attention Module) implementation."""

import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """CBAM-style Attention Module combining Channel Attention and Spatial Attention.

    References:
        Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018).
        CBAM: Convolutional Block Attention Module. In ECCV 2018.
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        """Initialize AttentionModule.

        Args:
            in_channels: Number of input feature channels (e.g. 2048 for ResNet50 Layer4).
            reduction: Reduction ratio for the Channel Attention bottleneck.
        """
        super().__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Cache for spatial attention weights (for explainability inspection)
        self.last_spatial_weights: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying Channel Attention followed by Spatial Attention.

        Args:
            x: Feature map tensor of shape (batch, channels, height, width).

        Returns:
            Attention-scaled feature map tensor of identical shape.
        """
        batch_size, channels, _, _ = x.size()

        # Channel attention computation
        avg_out = self.channel_fc(self.avg_pool(x).view(batch_size, channels))
        max_out = self.channel_fc(self.max_pool(x).view(batch_size, channels))
        channel_weight = torch.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        x = x * channel_weight

        # Spatial attention computation
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_conv(torch.cat([spatial_avg, spatial_max], dim=1))
        self.last_spatial_weights = spatial_weight.detach()

        x = x * spatial_weight
        return x
