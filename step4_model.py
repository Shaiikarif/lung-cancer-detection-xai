# step4_model.py
"""
STEP 4: Build the AI Model

We use a pretrained ResNet50 as the backbone.
WHY?
- ResNet50 was trained on millions of images
- It already knows how to detect edges, textures, patterns
- We just fine-tune it for our lung cancer task
- This is called "Transfer Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# ATTENTION MODULE
# ============================================================
class AttentionModule(nn.Module):
    """
    Attention mechanism that helps the model focus on important
    regions of the CT scan.

    Think of it like a spotlight - it tells the model
    "pay more attention to THIS area of the image"
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        # Channel Attention: "Which feature maps are important?"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        # Spatial Attention: "Which spatial locations are important?"
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # Channel attention
        avg_out = self.channel_fc(self.avg_pool(x).view(batch_size, channels))
        max_out = self.channel_fc(self.max_pool(x).view(batch_size, channels))
        channel_weight = torch.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        x = x * channel_weight

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_weight

        return x


# ============================================================
# MAIN MODEL
# ============================================================
class LungCancerModel(nn.Module):
    """
    Complete model for lung cancer classification.

    Architecture:
    1. ResNet50 backbone (pretrained on ImageNet)
    2. Attention module (focuses on important regions)
    3. Custom classifier (3 classes: Normal, Benign, Malignant)
    """

    def __init__(self, num_classes=3, pretrained=True, dropout_rate=0.5):
        super().__init__()

        # Load pretrained ResNet50
        if pretrained:
            print("  Loading pretrained ResNet50 weights...")
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)

        # Get the number of features from the last layer
        self.feature_dim = self.backbone.fc.in_features  # 2048

        # Remove the original classifier
        self.backbone.fc = nn.Identity()

        # Add attention module
        self.attention = AttentionModule(self.feature_dim)

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # Storage for Grad-CAM (we'll use this later for explainability)
        self.feature_maps = None
        self.gradients = None

    def _save_gradient(self, grad):
        """Hook to save gradients for Grad-CAM"""
        self.gradients = grad

    def forward(self, x):
        """
        Forward pass through the model.

        Input: Image tensor of shape (batch_size, 3, 224, 224)
        Output: Class logits of shape (batch_size, 3)
        """
        # Extract features using ResNet50 backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Save feature maps for Grad-CAM
        self.feature_maps = x
        if x.requires_grad:
            x.register_hook(self._save_gradient)

        # Apply attention
        x = self.attention(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 2048)

        # Classification
        output = self.classifier(x)

        return output


# ============================================================
# MODEL FACTORY
# ============================================================
def create_model(num_classes=3, pretrained=True):
    """
    Create and return the model with parameter count info.
    """

    print("\n" + "=" * 50)
    print("Building Model")
    print("=" * 50)

    model = LungCancerModel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=0.5
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Model: ResNet50 + Attention + Custom Classifier")
    print(f"  Total Parameters:     {total_params:>12,}")
    print(f"  Trainable Parameters: {trainable_params:>12,}")
    print(f"  Non-trainable:        {total_params - trainable_params:>12,}")
    print(f"  Number of classes:    {num_classes}")

    return model


# ============================================================
# TEST THE MODEL
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: MODEL BUILDING")
    print("=" * 60)

    # Create model
    model = create_model(num_classes=3, pretrained=True)

    # Test with a dummy input
    print("\n--- Testing model with dummy input ---")
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (raw logits): {output[0].tolist()}")

    # Apply softmax to get probabilities
    probs = F.softmax(output, dim=1)
    print(f"  Probabilities: {probs[0].tolist()}")

    # Get predictions
    predictions = torch.argmax(probs, dim=1)
    class_names = ['Normal', 'Benign', 'Malignant']
    print(f"  Predictions: {[class_names[p] for p in predictions.tolist()]}")

    print("\n✓ Step 4 complete! Model is working correctly.")
    print("  Proceed to Step 5: Training")
