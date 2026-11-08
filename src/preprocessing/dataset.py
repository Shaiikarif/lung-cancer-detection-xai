"""Dataset and DataLoader module for Lung Cancer CT scans."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.transforms import get_train_transforms, get_eval_transforms
from src.utils.logger import get_logger

logger = get_logger("dataset")


class LungCancerDataset(Dataset):
    """PyTorch Dataset for loading lung cancer CT scan images.

    Expected directory structure:
        <root_dir>/
            <phase>/
                normal/
                benign/
                malignant/
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "dataset",
        phase: str = "train",
        img_size: int = 224,
        classes: Optional[List[str]] = None,
        transform=None
    ) -> None:
        """Initialize LungCancerDataset.

        Args:
            root_dir: Base directory containing phase folders ('train', 'val', 'test').
            phase: 'train', 'val', or 'test'.
            img_size: Target image height and width.
            classes: List of class directory names (case-insensitive). Defaults to normal, benign, malignant.
            transform: Optional custom albumentations transform.
        """
        self.root_dir = Path(root_dir)
        self.phase = phase.lower()
        self.img_size = img_size

        if classes is None:
            self.classes = ["normal", "benign", "malignant"]
        else:
            self.classes = [c.lower() for c in classes]

        self.class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class: Dict[int, str] = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.images: List[str] = []
        self.labels: List[int] = []

        phase_dir = self.root_dir / self.phase

        if not phase_dir.exists():
            logger.warning("Phase directory does not exist: %s", phase_dir)
        else:
            valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            for class_name in self.classes:
                class_dir = phase_dir / class_name
                if not class_dir.exists():
                    logger.warning("Class directory missing: %s", class_dir)
                    continue

                for file in sorted(os.listdir(class_dir)):
                    if file.lower().endswith(valid_exts):
                        self.images.append(str(class_dir / file))
                        self.labels.append(self.class_to_idx[class_name])

        if transform is not None:
            self.transform = transform
        elif self.phase == "train":
            self.transform = get_train_transforms(img_size=self.img_size)
        else:
            self.transform = get_eval_transforms(img_size=self.img_size)

        logger.info(
            "[%s] Loaded %d images across %d classes",
            self.phase.upper(), len(self.images), len(self.classes)
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Fetch image, numeric label, and file path.

        Returns:
            image_tensor: Normalized torch.Tensor of shape (3, H, W).
            label: Integer label index (0, 1, or 2).
            img_path: Original absolute/relative file path string.
        """
        img_path = self.images[idx]
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Could not load image file: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image_tensor = augmented["image"]
        label = self.labels[idx]

        return image_tensor, label, img_path


def create_data_loaders(
    data_dir: Union[str, Path] = "dataset",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test PyTorch DataLoaders.

    Args:
        data_dir: Root dataset folder containing train/val/test subdirectories.
        batch_size: Batch size per iteration.
        img_size: Image size for resizing.
        num_workers: Number of background worker processes.
        pin_memory: Whether to pin memory for fast CUDA transfers.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = LungCancerDataset(data_dir, phase="train", img_size=img_size)
    val_dataset = LungCancerDataset(data_dir, phase="val", img_size=img_size)
    test_dataset = LungCancerDataset(data_dir, phase="test", img_size=img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
