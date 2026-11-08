# step3_data_pipeline.py
"""
STEP 3: Data loading and preprocessing pipeline
This file handles loading images, applying augmentations, and creating data loaders
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# ============================================================
# DATASET CLASS
# ============================================================
class LungCancerDataset(Dataset):
    """
    Custom Dataset for loading lung cancer CT scan images.

    What this does:
    - Reads images from folders (normal/benign/malignant)
    - Applies image transformations (resize, augment, normalize)
    - Returns image tensor + label for training
    """

    def __init__(self, root_dir, phase='train', img_size=224):
        """
        Args:
            root_dir: Path to dataset folder (e.g., 'dataset')
            phase: 'train', 'val', or 'test'
            img_size: Size to resize images to (224 for most models)
        """
        self.root_dir = root_dir
        self.phase = phase
        self.img_size = img_size

        # Define class names and their numeric labels
        self.classes = ['normal', 'benign', 'malignant']
        self.class_to_idx = {'normal': 0, 'benign': 1, 'malignant': 2}

        # Collect all image paths and labels
        self.images = []  # List of file paths
        self.labels = []  # List of numeric labels (0, 1, 2)

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, phase, class_name)

            if not os.path.exists(class_dir):
                print(f"  WARNING: {class_dir} does not exist!")
                continue

            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

        # Set up data augmentation transforms
        self.transform = self._get_transforms()

        # Print info
        print(f"  [{phase.upper()}] Loaded {len(self.images)} images")
        for cls in self.classes:
            count = self.labels.count(self.class_to_idx[cls])
            print(f"    - {cls}: {count}")

    def _get_transforms(self):
        """
        Define image transformations.

        Training: Random augmentations to prevent overfitting
        Validation/Test: Only resize and normalize
        """
        if self.phase == 'train':
            return A.Compose([
                # Resize all images to same size
                A.Resize(self.img_size, self.img_size),

                # Random flips (CT scans can be mirrored)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),

                # Random rotation
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=0.5
                ),

                # Color/brightness augmentations
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1
                    ),
                    A.CLAHE(clip_limit=4.0, p=1),
                ], p=0.4),

                # Normalize (ImageNet statistics - required for pretrained models)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),

                # Convert to PyTorch tensor
                ToTensorV2()
            ])
        else:
            # Validation and Test: no augmentation, just resize and normalize
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    def __len__(self):
        """Returns the total number of images"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns one image and its label.
        This is called automatically by DataLoader.
        """
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Convert BGR (OpenCV format) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        augmented = self.transform(image=image)
        image_tensor = augmented['image']

        # Get label
        label = self.labels[idx]

        return image_tensor, label, img_path


# ============================================================
# DATA LOADER CREATION
# ============================================================
def create_data_loaders(data_dir='dataset', batch_size=32, img_size=224):
    """
    Creates train, validation, and test data loaders.

    What is a DataLoader?
    - It loads images in batches (e.g., 32 at a time)
    - It shuffles training data each epoch
    - It loads data in parallel using multiple CPU workers
    """

    print("\n" + "=" * 50)
    print("Creating Data Loaders")
    print("=" * 50)

    # Create datasets
    train_dataset = LungCancerDataset(data_dir, phase='train', img_size=img_size)
    val_dataset = LungCancerDataset(data_dir, phase='val', img_size=img_size)
    test_dataset = LungCancerDataset(data_dir, phase='test', img_size=img_size)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # Shuffle training data
        num_workers=2,       # Use 2 CPU workers for loading
        pin_memory=True      # Faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,       # Don't shuffle validation data
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"\n  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }


# ============================================================
# VISUALIZATION
# ============================================================
def visualize_samples(dataset, num_samples=12, save_path='outputs/sample_images.png'):
    """
    Display sample images from the dataset to verify everything loaded correctly.
    """

    print("\nVisualizing sample images...")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    class_names = ['Normal', 'Benign', 'Malignant']
    class_colors = ['green', 'orange', 'red']

    # ImageNet denormalization values
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for i, ax in enumerate(axes.flat):
        if i < len(indices):
            img_tensor, label, path = dataset[indices[i]]

            # Denormalize for display
            img = img_tensor * std + mean
            img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            ax.set_title(
                f"{class_names[label]}",
                fontsize=14,
                color=class_colors[label],
                fontweight='bold'
            )
        ax.axis('off')

    plt.suptitle("Sample CT Scan Images from Dataset", fontsize=18, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {save_path}")
    plt.show()


def visualize_augmentations(dataset, idx=0, num_augments=8,
                             save_path='outputs/augmentations.png'):
    """
    Show how augmentation changes the same image differently each time.
    """

    print("\nVisualizing data augmentations...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    class_names = ['Normal', 'Benign', 'Malignant']

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, ax in enumerate(axes.flat):
        img_tensor, label, path = dataset[idx]

        img = img_tensor * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(f"Augmentation {i+1}", fontsize=12)
        ax.axis('off')

    plt.suptitle(
        f"Same Image with Different Augmentations (Class: {class_names[label]})",
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {save_path}")
    plt.show()


# ============================================================
# RUN THIS STEP
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: DATA PIPELINE")
    print("=" * 60)

    # Create data loaders
    loaders, datasets = create_data_loaders(
        data_dir='dataset',
        batch_size=32,
        img_size=224
    )

    # Visualize samples
    visualize_samples(datasets['train'])

    # Visualize augmentations
    visualize_augmentations(datasets['train'])

    # Test one batch
    print("\n--- Testing one batch ---")
    images, labels, paths = next(iter(loaders['train']))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels: {labels.tolist()[:10]}...")
    print(f"  Label dtype: {labels.dtype}")

    print("\n✓ Step 3 complete! Data pipeline is ready.")
    print("  Proceed to Step 4: Model Building")
