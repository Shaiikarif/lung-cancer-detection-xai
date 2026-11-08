# step5_train.py
"""
STEP 5: Train the Model

This is the most important step. The model learns to classify
CT scans by looking at thousands of examples.

Estimated time:
- GPU: 15-30 minutes
- CPU: 2-4 hours
"""

import os
import sys

# Fix Windows console encoding for Unicode characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our custom modules
from step3_data_pipeline import create_data_loaders
from step4_model import create_model


# ============================================================
# FOCAL LOSS (handles class imbalance)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss gives MORE weight to hard-to-classify examples
    and LESS weight to easy examples.

    This is important because our dataset might have unequal
    numbers of Normal, Benign, and Malignant images.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Why? To prevent overfitting.
    If the model keeps training after it stops improving on validation,
    it will memorize the training data instead of learning patterns.
    """

    def __init__(self, patience=10, path='checkpoints/best_model.pth'):
        self.patience = patience  # How many epochs to wait
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def check(self, val_loss, model):
        if val_loss < self.best_loss:
            # Improvement found!
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save(model.state_dict(), self.path)
            return True  # Improved
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # Not improved


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""

    model.train()  # Set model to training mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Progress bar
    pbar = tqdm(dataloader, desc='  Training', leave=False)

    for images, labels, _ in pbar:
        # Move data to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()       # Clear previous gradients
        outputs = model(images)     # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss

        # Backward pass
        loss.backward()             # Calculate gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()            # Update weights

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1


# ============================================================
# VALIDATION FUNCTION
# ============================================================
@torch.no_grad()  # Disable gradient computation for efficiency
def validate(model, dataloader, criterion, device):
    """Evaluate the model on validation data"""

    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels, _ in tqdm(dataloader, desc='  Validating', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels, all_probs


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_training_history(history, save_path='outputs/training_history.png'):
    """Plot training curves"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2, color='#3498db')
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2, color='#e74c3c')
    axes[0].set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2, color='#3498db')
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2, color='#e74c3c')
    axes[1].set_title('Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # F1 Score
    axes[2].plot(history['train_f1'], label='Train', linewidth=2, color='#3498db')
    axes[2].plot(history['val_f1'], label='Validation', linewidth=2, color='#e74c3c')
    axes[2].set_title('F1 Score Over Epochs', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend(fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Training history saved to {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path='outputs/confusion_matrix.png'):
    """Plot confusion matrix"""

    class_names = ['Normal', 'Benign', 'Malignant']
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], annot_kws={'size': 14})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], annot_kws={'size': 14})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Confusion matrix saved to {save_path}")
    plt.show()


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================
def train_model():
    """Complete training pipeline"""

    # ==================== CONFIGURATION ====================
    # You can modify these settings
    CONFIG = {
        'data_dir': 'dataset',
        'batch_size': 32,       # Reduce to 16 if you run out of memory
        'img_size': 224,
        'num_epochs': 30,       # 30-50 epochs is good
        'learning_rate': 0.0001,  # 1e-4
        'weight_decay': 0.0001,
        'patience': 10,         # Stop if no improvement for 10 epochs
        'num_classes': 3,
    }
    # ======================================================

    print("\n" + "=" * 60)
    print("STEP 5: TRAINING THE MODEL")
    print("=" * 60)

    # 1. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ Training on CPU will be SLOW. Consider using Google Colab for GPU.")

    # 2. Load data
    loaders, datasets = create_data_loaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size']
    )

    # 3. Create model
    model = create_model(num_classes=CONFIG['num_classes'], pretrained=True)
    model = model.to(device)

    # 4. Loss function (Focal Loss with class weights)
    # Give higher weight to malignant (most important to detect)
    class_weights = torch.tensor([1.0, 1.5, 2.0]).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    print(f"\n  Loss: Focal Loss (weights: Normal=1.0, Benign=1.5, Malignant=2.0)")

    # 5. Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    print(f"  Optimizer: AdamW (lr={CONFIG['learning_rate']})")

    # 6. Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 7. Early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG['patience'],
        path='checkpoints/best_model.pth'
    )

    # 8. Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }

    # ==================== TRAINING LOOP ====================
    print(f"\n{'='*60}")
    print(f"Starting Training: {CONFIG['num_epochs']} epochs")
    print(f"{'='*60}")

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(CONFIG['num_epochs']):
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 40)

        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, loaders['train'], criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, _, _, _ = validate(
            model, loaders['val'], criterion, device
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Check for improvement
        improved = early_stopping.check(val_loss, model)
        if improved:
            best_val_acc = val_acc
            print(f"  ✓ New best model! Val Acc: {best_val_acc:.4f}")
        else:
            print(f"  No improvement ({early_stopping.counter}/{early_stopping.patience})")

        # Check early stopping
        if early_stopping.should_stop:
            print(f"\n  ⚠ Early stopping! No improvement for {CONFIG['patience']} epochs.")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}")

    # ==================== LOAD BEST MODEL ====================
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load('checkpoints/best_model.pth', weights_only=True))

    # ==================== TEST EVALUATION ====================
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = validate(
        model, loaders['test'], criterion, device
    )

    # Classification Report
    class_names = ['Normal', 'Benign', 'Malignant']
    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1 Score: {test_f1:.4f}")

    print("\n  Detailed Classification Report:")
    print("  " + "-" * 55)
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        digits=4
    )
    print(report)

    # ROC-AUC
    try:
        test_probs = np.array(test_probs)
        auc = roc_auc_score(test_labels, test_probs, multi_class='ovr')
        print(f"  ROC-AUC Score: {auc:.4f}")
    except Exception:
        auc = None

    # ==================== SAVE RESULTS ====================

    # Plot training history
    plot_training_history(history)

    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds)

    # Save metrics to file
    metrics = {
        'test_accuracy': float(test_acc),
        'test_f1_score': float(test_f1),
        'test_loss': float(test_loss),
        'roc_auc': float(auc) if auc else None,
        'best_val_accuracy': float(best_val_acc),
        'total_training_time_minutes': round(total_time / 60, 1),
        'epochs_trained': len(history['train_loss']),
        'config': CONFIG
    }

    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n  ✓ Metrics saved to outputs/metrics.json")

    # Save complete model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'metrics': metrics,
        'history': history
    }, 'checkpoints/final_model.pth')
    print(f"  ✓ Final model saved to checkpoints/final_model.pth")

    print(f"\n✓ Step 5 complete! Model trained and evaluated.")
    print(f"  Best Test Accuracy: {test_acc:.4f}")
    print(f"  Proceed to Step 6: Explainability")

    return model, history, metrics


# ============================================================
# RUN TRAINING
# ============================================================
if __name__ == "__main__":
    model, history, metrics = train_model()
