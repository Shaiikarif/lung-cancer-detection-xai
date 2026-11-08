"""Image transformation pipelines and normalization utilities."""

from typing import Tuple, List, Union
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFAULT_MEAN: List[float] = [0.485, 0.456, 0.406]
DEFAULT_STD: List[float] = [0.229, 0.224, 0.225]


def get_train_transforms(
    img_size: int = 224,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD
) -> A.Compose:
    """Return albumentations transform pipeline for training with data augmentation."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=1.0
            ),
        ], p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                p=1.0
            ),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.4),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_eval_transforms(
    img_size: int = 224,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD
) -> A.Compose:
    """Return albumentations transform pipeline for validation and testing (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def unnormalize_tensor(
    tensor: torch.Tensor,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD
) -> torch.Tensor:
    """Revert ImageNet normalization on a tensor of shape (C, H, W) or (B, C, H, W).

    Returns a tensor with pixel values clamped to [0.0, 1.0].
    """
    device = tensor.device
    mean_t = torch.tensor(mean, device=device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)

    unnorm = tensor * std_t + mean_t
    return torch.clamp(unnorm, 0.0, 1.0)


def normalize_tensor(
    tensor: torch.Tensor,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD
) -> torch.Tensor:
    """Apply ImageNet normalization on a tensor with values in [0.0, 1.0]."""
    device = tensor.device
    mean_t = torch.tensor(mean, device=device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)

    return (tensor - mean_t) / std_t
