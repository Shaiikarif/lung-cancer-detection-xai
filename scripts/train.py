"""Training pipeline script for Lung Cancer Detection model."""

import argparse
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet50 import create_model
from src.models.losses import FocalLoss
from src.preprocessing.dataset import create_data_loaders
from src.evaluation.metrics import compute_classification_metrics
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train")


def set_seed(seed: int = 42) -> None:
    """Set random seed across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping monitor based on validation loss."""

    def __init__(self, patience: int = 10, save_path: Path = Path("checkpoints/best_model.pth")) -> None:
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            logger.info("Validation loss improved. Saved checkpoint to %s", self.save_path)
            return True
        else:
            self.counter += 1
            logger.info("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train model for a single epoch."""
    model.train()
    running_loss = 0.0

    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate loss and accuracy on validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validation", leave=False):
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
    parser = argparse.ArgumentParser(description="Train PulmoAI ResNet50-CBAM Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["random_seed"])

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    epochs = args.epochs or cfg["training"]["num_epochs"]
    lr = args.lr or cfg["training"]["learning_rate"]

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=cfg["data"]["dataset_dir"],
        batch_size=batch_size,
        img_size=cfg["data"]["img_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"]
    )

    model = create_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout_rate=cfg["model"]["dropout_rate"]
    ).to(device)

    criterion = FocalLoss(gamma=cfg["training"]["loss"]["gamma"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["training"]["scheduler"]["T_0"],
        T_mult=cfg["training"]["scheduler"]["T_mult"],
        eta_min=cfg["training"]["scheduler"]["eta_min"]
    )

    best_checkpoint = Path(cfg["checkpoints"]["dir"]) / cfg["checkpoints"]["best_model_filename"]
    early_stopping = EarlyStopping(
        patience=cfg["training"]["early_stopping"]["patience"],
        save_path=best_checkpoint
    )

    logger.info("Starting training for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch [%02d/%02d] -> Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%%",
            epoch, epochs, train_loss, val_loss, val_acc * 100
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # Save final model
    final_checkpoint = Path(cfg["checkpoints"]["dir"]) / cfg["checkpoints"]["final_model_filename"]
    torch.save(model.state_dict(), final_checkpoint)
    logger.info("Training complete. Final weights saved to %s", final_checkpoint)


if __name__ == "__main__":
    main()
