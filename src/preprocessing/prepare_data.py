"""Dataset organization and stratified train/val/test split utility."""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import Counter

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("prepare_data")


def organize_dataset(
    raw_dataset_path: str = "The IQ-OTHNCCD lung cancer dataset",
    output_dir: str = "dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> bool:
    """Split raw downloaded Kaggle dataset into stratified train, val, and test splits.

    Args:
        raw_dataset_path: Folder containing raw class subfolders.
        output_dir: Destination folder for train/val/test splits.
        train_ratio: Proportion of images allocated for training.
        val_ratio: Proportion of images allocated for validation.
        test_ratio: Proportion of images allocated for testing.
        random_seed: Seed for reproducibility.

    Returns:
        True if partitioning succeeded, False otherwise.
    """
    random.seed(random_seed)
    raw_path = Path(raw_dataset_path)
    out_path = Path(output_dir)

    folder_mapping = {
        "Normal cases": "normal",
        "Benign cases": "benign",
        "Malignant cases": "malignant",
        "normal": "normal",
        "benign": "benign",
        "malignant": "malignant",
        "Normal": "normal",
        "Benign": "benign",
        "Malignant": "malignant",
    }

    if not raw_path.exists():
        logger.error("Raw dataset directory not found at: %s", raw_path)
        return False

    logger.info("Found raw dataset at: %s", raw_path)

    for split in ["train", "val", "test"]:
        for cls in ["normal", "benign", "malignant"]:
            (out_path / split / cls).mkdir(parents=True, exist_ok=True)

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    total_stats: Dict[str, Dict[str, int]] = {}

    for folder_name in os.listdir(raw_path):
        folder_full_path = raw_path / folder_name
        if not folder_full_path.is_dir():
            continue

        class_name = folder_mapping.get(folder_name)
        if class_name is None:
            logger.warning("Unrecognized folder '%s', skipping...", folder_name)
            continue

        images = [
            f for f in os.listdir(folder_full_path)
            if f.lower().endswith(valid_extensions)
        ]

        if not images:
            for sub in os.listdir(folder_full_path):
                sub_path = folder_full_path / sub
                if sub_path.is_dir():
                    images.extend([
                        os.path.join(sub, f) for f in os.listdir(sub_path)
                        if f.lower().endswith(valid_extensions)
                    ])

        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        total_stats[class_name] = {s: len(imgs) for s, imgs in splits.items()}

        for split_name, split_images in splits.items():
            dest_folder = out_path / split_name / class_name
            for img in split_images:
                src_file = folder_full_path / img
                dst_file = dest_folder / Path(img).name
                shutil.copy2(src_file, dst_file)

    logger.info("Dataset successfully organized into '%s':", output_dir)
    for cls, stats in total_stats.items():
        logger.info(
            "  Class %-10s -> Train: %d | Val: %d | Test: %d | Total: %d",
            cls, stats["train"], stats["val"], stats["test"],
            stats["train"] + stats["val"] + stats["test"]
        )

    return True


if __name__ == "__main__":
    cfg = load_config()
    organize_dataset(
        raw_dataset_path=cfg["data"]["raw_dir"],
        output_dir=cfg["data"]["dataset_dir"],
        train_ratio=cfg["data"]["train_split"],
        val_ratio=cfg["data"]["val_split"],
        test_ratio=cfg["data"]["test_split"],
        random_seed=cfg["project"]["random_seed"]
    )
