"""Command-line inference and explainability script."""

import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet50 import create_model
from src.robustness.input_validation import InputValidator
from src.robustness.uncertainty import UncertaintyQuantifier
from src.explainability.engine import ExplainabilityEngine
from src.preprocessing.transforms import DEFAULT_MEAN, DEFAULT_STD
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("predict_script")


def main():
    parser = argparse.ArgumentParser(description="Run PulmoAI CT Scan Prediction and Explainability")
    parser.add_argument("--image", type=str, required=True, help="Path to input CT scan image")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Directory to save explanation")
    parser.add_argument("--explain", action="store_true", help="Generate multi-method XAI explanation panel")
    parser.add_argument("--mc-samples", type=int, default=30, help="Number of Monte Carlo dropout samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Input Validation
    validation = InputValidator.validate_file_path(args.image)
    if not validation.is_valid:
        logger.error("Validation failed: %s", validation.error_message)
        sys.exit(1)

    logger.info("Input validation passed: %dx%d (%d channels)", validation.width, validation.height, validation.channels)

    # Step 2: Load Model
    ckpt_path = Path(args.checkpoint or (Path(cfg["checkpoints"]["dir"]) / cfg["checkpoints"]["best_model_filename"]))
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s", ckpt_path)
        sys.exit(1)

    model = create_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Step 3: Explainability Engine & Uncertainty
    xai = ExplainabilityEngine(
        model=model,
        device=device,
        class_names=cfg["data"]["class_names"]
    )
    uq = UncertaintyQuantifier(
        model=model,
        device=device,
        class_names=cfg["data"]["class_names"],
        uncertainty_threshold=cfg["robustness"]["uncertainty"]["uncertainty_threshold"],
        confidence_threshold=cfg["robustness"]["uncertainty"]["confidence_threshold"]
    )

    _, _, tensor = xai.load_image(args.image, img_size=cfg["data"]["img_size"])
    uq_results = uq.estimate(tensor, num_samples=args.mc_samples)

    print("\n" + "=" * 60)
    print("PULMOAI DIAGNOSTIC INFERENCE RESULT")
    print("=" * 60)
    print(f"File:               {Path(args.image).name}")
    print(f"Prediction:         {uq_results['predicted_class']}")
    print(f"Confidence:         {uq_results['confidence']:.2%}")
    print(f"Uncertainty (Std):  {uq_results['uncertainty']:.4f}")
    print(f"Entropy:            {uq_results['entropy']:.4f}")
    print(f"Reliability Status: {'RELIABLE' if uq_results['is_reliable'] else 'AMBIGUOUS / HIGH UNCERTAINTY'}")
    print("-" * 60)
    for idx, name in enumerate(cfg["data"]["class_names"]):
        prob = uq_results["mean_probs"][idx]
        std = uq_results["std_probs"][idx]
        print(f"  {name:<12s}: {prob:.2%} +/- {std:.2%}")
    print("=" * 60)

    if args.explain:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_plot = out_dir / f"{Path(args.image).stem}_explanation.png"
        xai.explain(args.image, save_path=save_plot, include_lime=False)
        logger.info("Visual explanation saved to: %s", save_plot)


if __name__ == "__main__":
    main()
