# step2_prepare_data.py
"""
STEP 2: Prepare and organize the dataset
Run this AFTER downloading the dataset from Kaggle
"""

import os
import shutil
import random
from collections import Counter

# ============================================================
# CONFIGURATION - UPDATE THIS PATH
# ============================================================
# Change this to wherever you extracted the Kaggle dataset
RAW_DATASET_PATH = "The IQ-OTHNCCD lung cancer dataset"  # UPDATE THIS!

# Output directory
OUTPUT_DIR = "dataset"

# Split ratios
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15    # 15% for validation
TEST_RATIO = 0.15   # 15% for testing

RANDOM_SEED = 42
# ============================================================


def organize_dataset():
    """Organize raw dataset into train/val/test splits"""

    random.seed(RANDOM_SEED)

    # Mapping of folder names in the downloaded dataset to our class names
    # The Kaggle dataset might have different folder names
    # Check your downloaded folder and update if needed
    folder_mapping = {
        "Normal cases": "normal",
        "Benign cases": "benign",
        "Malignant cases": "malignant",
        # Alternative names (in case the dataset uses different names)
        "normal": "normal",
        "benign": "benign",
        "malignant": "malignant",
        "Normal": "normal",
        "Benign": "benign",
        "Malignant": "malignant",
    }

    # Check if raw dataset exists
    if not os.path.exists(RAW_DATASET_PATH):
        print(f"ERROR: Dataset not found at '{RAW_DATASET_PATH}'")
        print(f"\nPlease update RAW_DATASET_PATH in this file.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"\nFiles in current directory:")
        for item in os.listdir('.'):
            print(f"  {item}")
        return False

    print(f"Found dataset at: {RAW_DATASET_PATH}")
    print(f"Contents: {os.listdir(RAW_DATASET_PATH)}")

    # Create output directories
    for split in ['train', 'val', 'test']:
        for cls in ['normal', 'benign', 'malignant']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    total_stats = {}

    # Process each class
    for folder_name in os.listdir(RAW_DATASET_PATH):
        folder_path = os.path.join(RAW_DATASET_PATH, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Get class name
        class_name = folder_mapping.get(folder_name)
        if class_name is None:
            print(f"  WARNING: Unknown folder '{folder_name}', skipping...")
            continue

        # Get all image files
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        images = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_extensions)
        ]

        if len(images) == 0:
            # Check subdirectories
            for sub in os.listdir(folder_path):
                sub_path = os.path.join(folder_path, sub)
                if os.path.isdir(sub_path):
                    images.extend([
                        os.path.join(sub, f) for f in os.listdir(sub_path)
                        if f.lower().endswith(valid_extensions)
                    ])

        print(f"\n  Class '{class_name}': Found {len(images)} images")

        if len(images) == 0:
            print(f"  WARNING: No images found in {folder_path}")
            continue

        # Shuffle images
        random.shuffle(images)

        # Calculate split sizes
        total = len(images)
        train_size = int(total * TRAIN_RATIO)
        val_size = int(total * VAL_RATIO)
        test_size = total - train_size - val_size

        # Split
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        # Copy files
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        for split_name, split_images in splits.items():
            for img_name in split_images:
                src = os.path.join(folder_path, img_name)
                # Handle case where img_name might include subdirectory
                dst_name = os.path.basename(img_name)
                dst = os.path.join(OUTPUT_DIR, split_name, class_name, dst_name)

                if os.path.exists(src):
                    shutil.copy2(src, dst)

        total_stats[class_name] = {
            'total': total,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        }

        print(f"    Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"{'Class':<15} {'Total':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 55)

    for cls, stats in total_stats.items():
        print(f"{cls:<15} {stats['total']:<10} {stats['train']:<10} "
              f"{stats['val']:<10} {stats['test']:<10}")

    print("-" * 55)
    total_all = sum(s['total'] for s in total_stats.values())
    total_train = sum(s['train'] for s in total_stats.values())
    total_val = sum(s['val'] for s in total_stats.values())
    total_test = sum(s['test'] for s in total_stats.values())
    print(f"{'TOTAL':<15} {total_all:<10} {total_train:<10} "
          f"{total_val:<10} {total_test:<10}")

    print(f"\n✓ Dataset prepared successfully in '{OUTPUT_DIR}/' folder!")
    print(f"  You can now proceed to Step 3.")

    return True


def verify_dataset():
    """Verify the dataset is properly organized"""

    print("\n" + "=" * 60)
    print("VERIFYING DATASET")
    print("=" * 60)

    all_good = True

    for split in ['train', 'val', 'test']:
        print(f"\n  [{split.upper()}]")
        for cls in ['normal', 'benign', 'malignant']:
            path = os.path.join(OUTPUT_DIR, split, cls)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                status = "✓" if count > 0 else "✗"
                print(f"    {status} {cls}: {count} images")
                if count == 0:
                    all_good = False
            else:
                print(f"    ✗ {cls}: DIRECTORY NOT FOUND")
                all_good = False

    if all_good:
        print(f"\n  ✓ Dataset is ready!")
    else:
        print(f"\n  ✗ Dataset has issues. Please check the paths.")

    return all_good


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: DATASET PREPARATION")
    print("=" * 60)

    success = organize_dataset()

    if success:
        verify_dataset()
