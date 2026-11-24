"""
PulmoAI Clean Retraining Pipeline.
Trains ResNet-50 + CBAM Attention network on certified leak-free clean splits.
Saves all artifacts strictly inside research_audit/checkpoints/ to preserve
historical baseline checkpoints untouched.
"""

import os
import sys
import json
import random
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet50 import create_model
from src.models.losses import FocalLoss
from src.preprocessing.dataset import LungCancerDataset
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train_clean")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience: int = 8, save_path: Path = Path("research_audit/checkpoints/retrained_best_model.pth")):
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int, model: nn.Module) -> bool:
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            logger.info("Validation loss improved to %.4f. Saved checkpoint to %s", val_loss, self.save_path)
            return True
        else:
            self.counter += 1
            logger.info("EarlyStopping counter: %d / %d (Best was %.4f at epoch %d)", self.counter, self.patience, self.best_loss, self.best_epoch)
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (VRAM: %.2f GB)", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)

    clean_data_dir = PROJECT_ROOT / "research_audit" / "clean_splits" / "data"
    ckpt_dir = PROJECT_ROOT / "research_audit" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16  # Stable for RTX 3050 Laptop 4GB VRAM
    num_workers = 0  # Avoid Windows IPC overhead
    epochs = 20
    lr = 5e-5

    train_dataset = LungCancerDataset(clean_data_dir, phase="train", img_size=224)
    val_dataset = LungCancerDataset(clean_data_dir, phase="val", img_size=224)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info("Initialized Clean DataLoaders: Train=%d batches (%d samples), Val=%d batches (%d samples)",
                len(train_loader), len(train_dataset), len(val_loader), len(val_dataset))

    # Create model: ResNet50 + CBAM
    model = create_model(num_classes=3, pretrained=True, dropout_rate=0.5).to(device)

    best_checkpoint_path = ckpt_dir / "retrained_best_model.pth"
    final_checkpoint_path = ckpt_dir / "retrained_final_model.pth"

    # If best checkpoint already exists from earlier epochs, load it as initialization
    initial_best_loss = float("inf")
    if best_checkpoint_path.exists():
        logger.info("Loading existing retrained weights from %s...", best_checkpoint_path)
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        # Evaluate current loss
        criterion_eval = FocalLoss(gamma=2.0)
        init_loss, init_acc = evaluate_epoch(model, val_loader, criterion_eval, device)
        logger.info("Pre-loaded model Validation Loss: %.4f | Accuracy: %.2f%%", init_loss, init_acc * 100)
        initial_best_loss = init_loss

    # Focal loss (gamma=2.0)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=5, save_path=best_checkpoint_path)
    if initial_best_loss < float("inf"):
        early_stopping.best_loss = initial_best_loss

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_times": []
    }

    start_time = time.time()
    logger.info("Starting clean retraining for up to %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_dur = time.time() - t0

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(cur_lr))
        history["epoch_times"].append(round(epoch_dur, 2))

        logger.info(
            "Epoch [%02d/%02d] (%.1fs) | Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%% | LR: %.2e",
            epoch, epochs, epoch_dur, train_loss, train_acc * 100, val_loss, val_acc * 100, cur_lr
        )

        improved = early_stopping(val_loss, epoch, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered at epoch %d. Best model was saved at epoch %d with val_loss %.4f",
                        epoch, early_stopping.best_epoch, early_stopping.best_loss)
            break

    total_duration = time.time() - start_time
    # Save final model
    torch.save(model.state_dict(), final_checkpoint_path)
    logger.info("Retraining complete in %.1f seconds. Final checkpoint saved to %s", total_duration, final_checkpoint_path)

    # Save training history JSON
    history_meta = {
        "architecture": "ResNet50 + CBAM Attention",
        "loss_function": "FocalLoss(gamma=2.0)",
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
        "scheduler": "CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)",
        "batch_size": batch_size,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_training_time_seconds": round(total_duration, 2),
        "best_epoch": early_stopping.best_epoch,
        "best_val_loss": round(early_stopping.best_loss, 4),
        "epochs_trained": len(history["train_loss"]),
        "history": history
    }

    history_json_path = ckpt_dir / "training_history.json"
    with open(history_json_path, 'w', encoding='utf-8') as f:
        json.dump(history_meta, f, indent=4)
    logger.info("Training history saved to %s", history_json_path)

    # Generate Training History Curve Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.axvline(x=early_stopping.best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({early_stopping.best_epoch})")
    ax1.set_title("Loss Curves (Focal Loss)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#1f77b4", linewidth=2)
    ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#ff7f0e", linewidth=2)
    ax2.axvline(x=early_stopping.best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({early_stopping.best_epoch})")
    ax2.set_title("Accuracy Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = ckpt_dir / "training_history.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Training history plot saved to %s", plot_path)


if __name__ == "__main__":
    main()
