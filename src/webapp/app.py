"""PulmoAI Web Application Flask module."""

import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet50 import create_model
from src.robustness.input_validation import InputValidator
from src.robustness.uncertainty import UncertaintyQuantifier
from src.explainability.engine import ExplainabilityEngine
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("webapp")


def create_app(config_path: str = None) -> Flask:
    """Application factory for PulmoAI web interface."""
    cfg = load_config(config_path)

    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static")
    )

    upload_folder = PROJECT_ROOT / cfg["webapp"]["upload_folder"]
    results_folder = PROJECT_ROOT / cfg["webapp"]["results_folder"]
    samples_folder = PROJECT_ROOT / cfg["webapp"]["samples_folder"]

    upload_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)
    samples_folder.mkdir(parents=True, exist_ok=True)

    app.config["UPLOAD_FOLDER"] = str(upload_folder)
    app.config["RESULTS_FOLDER"] = str(results_folder)
    app.config["SAMPLES_FOLDER"] = str(samples_folder)
    app.config["MAX_CONTENT_LENGTH"] = cfg["webapp"]["max_content_length_mb"] * 1024 * 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Web App on device: %s", device)

    # Load Model
    model = create_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False
    ).to(device)

    ckpt_path = PROJECT_ROOT / cfg["checkpoints"]["dir"] / cfg["checkpoints"]["best_model_filename"]
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info("Model weights loaded successfully from %s", ckpt_path)
    else:
        logger.warning("Checkpoint not found at %s! Running with uninitialized weights.", ckpt_path)

    model.eval()

    xai_engine = ExplainabilityEngine(
        model=model,
        device=device,
        class_names=cfg["data"]["class_names"],
        mean=cfg["data"]["normalization"]["mean"],
        std=cfg["data"]["normalization"]["std"]
    )

    uq_quantifier = UncertaintyQuantifier(
        model=model,
        device=device,
        class_names=cfg["data"]["class_names"],
        uncertainty_threshold=cfg["robustness"]["uncertainty"]["uncertainty_threshold"],
        confidence_threshold=cfg["robustness"]["uncertainty"]["confidence_threshold"]
    )

    def process_ct_image(image_path: str) -> Dict[str, Any]:
        """Internal processing pipeline for CT scans."""
        rgb_uint8, rgb_float, tensor = xai_engine.load_image(
            image_path,
            img_size=cfg["data"]["img_size"]
        )

        uq_res = uq_quantifier.estimate(tensor, num_samples=cfg["robustness"]["uncertainty"]["mc_samples"])
        gcam_res = xai_engine.gradcam.explain(tensor, rgb_float, target_class=uq_res["predicted_class_idx"])

        pred_class = uq_res["predicted_class"]
        confidence = uq_res["confidence"]

        if pred_class == "Normal":
            risk_level = "Low"
            recommendation = "No prominent malignant lesion detected. Maintain routine clinical screening."
        elif pred_class == "Benign":
            risk_level = "Moderate"
            recommendation = "Benign pulmonary nodule or inflammatory manifestation. Follow-up CT in 6-12 months recommended."
        else:
            risk_level = "High"
            recommendation = "Malignant characteristics detected. Immediate pulmonary/oncological biopsy evaluation indicated."

        prob_dict = {
            cls_name: float(uq_res["mean_probs"][i])
            for i, cls_name in enumerate(cfg["data"]["class_names"])
        }

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "uncertainty": uq_res["uncertainty"],
            "entropy": uq_res["entropy"],
            "is_reliable": uq_res["is_reliable"],
            "overlay": gcam_res["overlay"]
        }

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "model": cfg["model"]["name"], "device": str(device)})

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Please select an image."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        val_result = InputValidator.validate_file_path(filepath)
        if not val_result.is_valid:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": val_result.error_message}), 400

        try:
            result = process_ct_image(filepath)
            gradcam_filename = f"gradcam_{Path(filename).stem}.png"
            gradcam_path = os.path.join(app.config["RESULTS_FOLDER"], gradcam_filename)

            overlay_img = Image.fromarray(result["overlay"])
            overlay_img.save(gradcam_path)

            return jsonify({
                "success": True,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "risk_level": result["risk_level"],
                "recommendation": result["recommendation"],
                "uncertainty": result["uncertainty"],
                "entropy": result["entropy"],
                "is_reliable": result["is_reliable"],
                "original_url": f"/static/uploads/{filename}",
                "gradcam_url": f"/static/results/{gradcam_filename}",
            })
        except Exception as e:
            logger.error("Inference error: %s", str(e), exc_info=True)
            return jsonify({"error": f"Internal processing error: {str(e)}"}), 500

    @app.route("/api/sample/<sample_type>", methods=["GET"])
    def analyze_sample(sample_type):
        sample_mapping = {
            "normal": os.path.join(app.config["SAMPLES_FOLDER"], "normal_sample.jpg"),
            "benign": os.path.join(app.config["SAMPLES_FOLDER"], "benign_sample.jpg"),
            "malignant": os.path.join(app.config["SAMPLES_FOLDER"], "malignant_sample.jpg"),
        }

        if sample_type not in sample_mapping:
            return jsonify({"error": "Invalid sample type. Choose normal, benign, or malignant."}), 400

        src_path = sample_mapping[sample_type]
        if not os.path.exists(src_path):
            return jsonify({"error": f"Sample file not found: {src_path}"}), 404

        dest_filename = f"{uuid.uuid4()}_sample_{sample_type}.jpg"
        dest_filepath = os.path.join(app.config["UPLOAD_FOLDER"], dest_filename)
        shutil.copy(src_path, dest_filepath)

        try:
            result = process_ct_image(dest_filepath)
            gradcam_filename = f"gradcam_{Path(dest_filename).stem}.png"
            gradcam_path = os.path.join(app.config["RESULTS_FOLDER"], gradcam_filename)

            overlay_img = Image.fromarray(result["overlay"])
            overlay_img.save(gradcam_path)

            return jsonify({
                "success": True,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "risk_level": result["risk_level"],
                "recommendation": result["recommendation"],
                "uncertainty": result["uncertainty"],
                "entropy": result["entropy"],
                "is_reliable": result["is_reliable"],
                "original_url": f"/static/uploads/{dest_filename}",
                "gradcam_url": f"/static/results/{gradcam_filename}",
            })
        except Exception as e:
            logger.error("Sample processing error: %s", str(e), exc_info=True)
            return jsonify({"error": f"Sample processing error: {str(e)}"}), 500

    return app
