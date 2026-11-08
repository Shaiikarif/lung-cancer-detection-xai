"""Evaluation pipeline coordinator."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics
from src.utils.logger import get_logger

logger = get_logger("evaluator")


class ModelEvaluator:
    """Runs full evaluation on a PyTorch DataLoader and produces summary metrics and plots."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.class_names = class_names or ["Normal", "Benign", "Malignant"]
        self.model.eval()

    def collect_predictions(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Iterate over dataset and collect all true labels, predictions, and probabilities.

        Returns:
            (y_true, y_pred, y_probs)
        """
        all_preds = []
        all_labels = []
        all_probs = []

        logger.info("Collecting test set predictions...")
        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Compute all metrics and optionally export plots and JSON reports."""
        y_true, y_pred, y_probs = self.collect_predictions(dataloader)
        results = compute_classification_metrics(y_true, y_pred, y_probs, self.class_names)

        overall = results["overall_metrics"]
        logger.info("Evaluation Complete:")
        logger.info("  Accuracy:  %.4f (%.2f%%)", overall["accuracy"], overall["accuracy"] * 100)
        logger.info("  Weighted F1: %.4f", overall["f1_weighted"])
        logger.info("  Macro F1:    %.4f", overall["f1_macro"])
        logger.info("  Micro ROC-AUC: %.4f", overall["roc_auc_micro"])

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Export JSON
            json_file = out_path / "evaluation_metrics.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            logger.info("Exported metrics JSON to %s", json_file)

            # Export Confusion Matrix Plot
            self.plot_confusion_matrix(y_true, y_pred, out_path / "eval_confusion_matrix.png")

        return results

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
        """Plot and save confusion matrix figure."""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names, yticklabels=self.class_names,
            ax=ax, annot_kws={"size": 14}
        )
        ax.set_title("Test Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix plot to %s", save_path)
