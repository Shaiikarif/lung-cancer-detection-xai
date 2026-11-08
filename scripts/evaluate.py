"""Evaluation script for trained PulmoAI checkpoint."""

import argparse
import json
import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet50 import create_model
from src.preprocessing.dataset import create_data_loaders
from src.evaluation.evaluator import ModelEvaluator
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate_script")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PulmoAI Lung Cancer Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save eval plots and metrics")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    ckpt_path = Path(args.checkpoint or (Path(cfg["checkpoints"]["dir"]) / cfg["checkpoints"]["best_model_filename"]))
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s", ckpt_path)
        sys.exit(1)

    # Load DataLoaders
    _, _, test_loader = create_data_loaders(
        data_dir=cfg["data"]["dataset_dir"],
        batch_size=cfg["training"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=False
    )

    # Load Model
    model = create_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False
    ).to(device)

    logger.info("Loading checkpoint from %s...", ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    evaluator = ModelEvaluator(
        model=model,
        device=device,
        class_names=cfg["data"]["class_names"]
    )

    results = evaluator.evaluate(test_loader, output_dir=args.output_dir)
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy:         {results['overall_metrics']['accuracy']:.4f}")
    print(f"Weighted F1:      {results['overall_metrics']['f1_weighted']:.4f}")
    print(f"Macro F1:         {results['overall_metrics']['f1_macro']:.4f}")
    print(f"Micro ROC-AUC:    {results['overall_metrics']['roc_auc_micro']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
