# step8_webapp.py
"""
STEP 8: Web Application (FIXED & COMPLETE VERSION)

This creates a web interface where users can:
1. Upload a CT scan image
2. Get AI prediction with confidence
3. See Grad-CAM explanation
4. See uncertainty analysis

Run this and open http://localhost:5000 in your browser
"""

import os
import sys
import uuid

# Fix Windows console encoding for Unicode characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from step4_model import create_model

# ============================================================
# APP SETUP
# ============================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}")

model = create_model(num_classes=3, pretrained=False)
checkpoint_path = 'checkpoints/best_model.pth'

if os.path.exists(checkpoint_path):
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    print("  ✓ Model loaded successfully!")
else:
    print("  ⚠ No checkpoint found! Using random weights.")
    print(f"    Expected: {checkpoint_path}")
    print(f"    Run step5_train.py first!")

model = model.to(device)
model.eval()

# Constants
CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
IMG_SIZE = 224


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def allowed_file(filename):
    """Check if file has valid image extension"""
    valid_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in valid_extensions


def load_and_preprocess(filepath):
    """Load image and prepare for model input"""

    # Read image
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Cannot read image: {filepath}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # For display (0 to 1 range)
    img_display = img.astype(np.float32) / 255.0

    # For model (normalized with ImageNet stats)
    img_normalized = (img_display - MEAN) / STD
    img_tensor = torch.from_numpy(
        img_normalized.transpose(2, 0, 1)  # HWC -> CHW
    ).float().unsqueeze(0).to(device)       # Add batch dimension

    return img, img_display, img_tensor


def get_prediction(img_tensor):
    """Get model prediction"""

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    return pred_class, confidence, probs


def generate_gradcam(img_tensor, img_display, pred_class):
    """Generate Grad-CAM heatmap overlay"""

    model.eval()

    # Target layer: last conv layer of ResNet50
    target_layer = model.backbone.layer4[-1]

    # Create Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_class)]

    # Generate heatmap
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]

    # Overlay on image
    overlay = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

    # ✅ FIX: Use del instead of cam.release()
    del cam

    return overlay, grayscale_cam


def estimate_uncertainty(img_tensor, num_passes=20):
    """MC Dropout uncertainty estimation"""

    # Enable ONLY dropout layers (keep batchnorm in eval mode)
    def enable_dropout_only(m):
        for module in m.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    model.eval()
    enable_dropout_only(model)

    predictions = []
    for _ in range(num_passes):
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
            predictions.append(probs)

    # Back to full eval mode
    model.eval()

    predictions = np.array(predictions)  # (num_passes, 3)

    mean_probs = predictions.mean(axis=0)
    std_probs = predictions.std(axis=0)

    pred_class = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_class])
    uncertainty = float(std_probs[pred_class])
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-10)))

    is_reliable = (uncertainty < 0.15) and (confidence > 0.7)

    return {
        'confidence': confidence,
        'uncertainty': uncertainty,
        'entropy': entropy,
        'is_reliable': is_reliable
    }


def process_image(filepath):
    """
    Complete image processing pipeline:
    1. Load & preprocess image
    2. Get prediction
    3. Generate Grad-CAM
    4. Estimate uncertainty
    """

    # Load image
    img, img_display, img_tensor = load_and_preprocess(filepath)

    # Get prediction
    pred_class, confidence, probs = get_prediction(img_tensor)

    # Generate Grad-CAM
    overlay, heatmap = generate_gradcam(img_tensor, img_display, pred_class)

    # Estimate uncertainty
    uncertainty_info = estimate_uncertainty(img_tensor)

    # Risk level and recommendation
    risk_map = {
        0: 'LOW',
        1: 'MODERATE',
        2: 'HIGH'
    }

    recommendation_map = {
        0: 'No significant abnormalities detected. Continue regular screening.',
        1: 'Benign nodule detected. Follow-up CT scan recommended in 3-6 months.',
        2: 'Suspicious malignant features detected. Immediate specialist consultation recommended.'
    }

    return {
        'prediction': CLASS_NAMES[pred_class],
        'confidence': float(confidence),
        'probabilities': {
            CLASS_NAMES[i]: float(probs[i]) for i in range(3)
        },
        'risk_level': risk_map[pred_class],
        'recommendation': recommendation_map[pred_class],
        'uncertainty': uncertainty_info['uncertainty'],
        'entropy': uncertainty_info['entropy'],
        'is_reliable': uncertainty_info['is_reliable'],
        'overlay': overlay,
    }


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction"""

    # Check if file exists in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Please select an image.'}), 400

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected. Please choose an image.'}), 400

    # Check file type
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or BMP.'
        }), 400

    try:
        # Save uploaded file
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"\n  Processing: {file.filename}")

        # Process image
        result = process_image(filepath)

        # Save Grad-CAM overlay image
        gradcam_filename = f'gradcam_{filename}'

        # Ensure filename ends with .png for saving
        if not gradcam_filename.lower().endswith('.png'):
            gradcam_filename = gradcam_filename.rsplit('.', 1)[0] + '.png'

        gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], gradcam_filename)

        # Save the overlay image
        overlay_image = Image.fromarray(result['overlay'])
        overlay_image.save(gradcam_path)

        print(f"  Prediction: {result['prediction']} ({result['confidence']:.1%})")
        print(f"  Reliable: {'Yes' if result['is_reliable'] else 'No'}")

        # Build response (don't include numpy arrays)
        response = {
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'risk_level': result['risk_level'],
            'recommendation': result['recommendation'],
            'uncertainty': result['uncertainty'],
            'entropy': result['entropy'],
            'is_reliable': result['is_reliable'],
            'original_url': f'/static/uploads/{filename}',
            'gradcam_url': f'/static/results/{gradcam_filename}',
        }

        return jsonify(response)

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/sample/<sample_type>', methods=['GET'])
def analyze_sample(sample_type):
    """Analyze a preset sample CT scan (normal, benign, malignant)"""
    sample_mapping = {
        'normal': 'static/samples/normal_sample.jpg',
        'benign': 'static/samples/benign_sample.jpg',
        'malignant': 'static/samples/malignant_sample.jpg'
    }

    if sample_type not in sample_mapping:
        return jsonify({'error': 'Invalid sample type. Choose normal, benign, or malignant.'}), 400

    source_path = sample_mapping[sample_type]
    if not os.path.exists(source_path):
        return jsonify({'error': f'Sample file not found: {source_path}'}), 404

    try:
        filename = f"{uuid.uuid4()}_sample_{sample_type}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        import shutil
        shutil.copy(source_path, filepath)

        result = process_image(filepath)

        gradcam_filename = f'gradcam_{filename}.png'
        gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], gradcam_filename)

        overlay_image = Image.fromarray(result['overlay'])
        overlay_image.save(gradcam_path)

        response = {
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'risk_level': result['risk_level'],
            'recommendation': result['recommendation'],
            'uncertainty': result['uncertainty'],
            'entropy': result['entropy'],
            'is_reliable': result['is_reliable'],
            'original_url': f'/static/uploads/{filename}',
            'gradcam_url': f'/static/results/{gradcam_filename}',
        }
        return jsonify(response)
    except Exception as e:
        print(f"  Sample error: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/symptoms', methods=['POST'])
def analyze_symptoms():
    """Calculate clinical symptom risk score"""
    try:
        data = request.get_json(force=True) or {}

        age = float(data.get('age', 50))
        smoking = float(data.get('smoking', 0))
        cough_blood = float(data.get('cough_blood', 0))
        chest_pain = float(data.get('chest_pain', 0))
        shortness_breath = float(data.get('shortness_breath', 0))
        fatigue = float(data.get('fatigue', 0))
        weight_loss = float(data.get('weight_loss', 0))
        swallowing_diff = float(data.get('swallowing_diff', 0))

        # Clinical Risk Scoring
        score = 0.0
        if age > 50: score += 15.0
        if age > 65: score += 10.0
        score += smoking * 20.0
        score += cough_blood * 30.0
        score += chest_pain * 12.0
        score += shortness_breath * 10.0
        score += fatigue * 8.0
        score += weight_loss * 15.0
        score += swallowing_diff * 10.0

        risk_percentage = min(99.0, max(2.0, score))

        if risk_percentage < 25.0:
            risk_level = 'LOW'
            recommendation = 'Low symptom-based risk. Maintain standard wellness checkups.'
        elif risk_percentage < 60.0:
            risk_level = 'MODERATE'
            recommendation = 'Moderate symptom risk. Schedule a clinical consultation and chest CT scan.'
        else:
            risk_level = 'HIGH'
            recommendation = 'High clinical symptom risk detected. Urgent pulmonology evaluation and CT imaging advised.'

        return jsonify({
            'success': True,
            'risk_percentage': round(risk_percentage, 1),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'symptom_factors': {
                'Age Factor': 'Elevated' if age > 50 else 'Normal',
                'Smoking History': 'Heavy' if smoking == 2 else ('Moderate' if smoking == 1 else 'None'),
                'Hemoptysis (Coughing Blood)': 'Present' if cough_blood > 0 else 'Absent',
                'Unexplained Weight Loss': 'Present' if weight_loss > 0 else 'Absent'
            }
        })
    except Exception as e:
        return jsonify({'error': f'Symptom evaluation error: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'device': str(device),
        'model_loaded': os.path.exists(checkpoint_path)
    })


# ============================================================
# CREATE HTML TEMPLATE
# ============================================================
def create_template():
    """Create the HTML template file if missing"""
    if os.path.exists('templates/index.html'):
        return

    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Cancer Detection - Explainable AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #0c0c1d 0%, #1a1a3e 50%, #0d0d2b 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }

        .header {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            padding: 25px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .header h1 { color: #00d4ff; font-size: 26px; margin-bottom: 5px; }
        .header p { color: #8892b0; font-size: 14px; }

        .container { max-width: 1200px; margin: 30px auto; padding: 0 20px; }

        /* Upload Area */
        .upload-area {
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(0,212,255,0.3);
            border-radius: 16px;
            padding: 50px 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #00d4ff;
            background: rgba(255,255,255,0.08);
            transform: translateY(-2px);
        }
        .upload-area h2 { color: #00d4ff; margin-bottom: 10px; font-size: 22px; }
        .upload-area p { color: #8892b0; margin-bottom: 20px; }

        .btn {
            background: linear-gradient(135deg, #00d4ff, #0077b6);
            color: white;
            border: none;
            padding: 14px 40px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,212,255,0.3);
        }
        .btn:active { transform: translateY(0); }

        #preview {
            max-width: 250px;
            max-height: 250px;
            margin-top: 20px;
            border-radius: 10px;
            border: 2px solid rgba(0,212,255,0.3);
            display: none;
        }

        /* Loading */
        .loading {
            display: none;
            text-align: center;
            padding: 50px;
        }
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #00d4ff;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading p { color: #8892b0; }

        /* Results */
        .results { display: none; margin-top: 30px; }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }

        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 14px;
            padding: 25px;
            transition: transform 0.2s;
        }
        .card:hover { transform: translateY(-3px); }
        .card h3 {
            color: #00d4ff;
            margin-bottom: 12px;
            font-size: 16px;
        }
        .card img {
            width: 100%;
            border-radius: 10px;
            margin-top: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card .desc {
            font-size: 12px;
            color: #8892b0;
            margin-top: 5px;
        }

        /* Diagnosis Card */
        .diagnosis-card {
            grid-column: 1 / -1;
            text-align: center;
            padding: 35px;
            background: rgba(255,255,255,0.07);
        }

        .diagnosis-text {
            font-size: 36px;
            font-weight: 800;
            margin: 10px 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .risk-low { color: #2ecc71; }
        .risk-moderate { color: #f39c12; }
        .risk-high { color: #e74c3c; }

        .confidence-text {
            font-size: 18px;
            margin: 5px 0 15px;
        }

        /* Badge */
        .badge {
            display: inline-block;
            padding: 6px 18px;
            border-radius: 25px;
            font-size: 13px;
            font-weight: 700;
            margin: 10px 0;
            letter-spacing: 0.5px;
        }
        .badge-ok {
            background: rgba(46,204,113,0.15);
            color: #2ecc71;
            border: 1.5px solid #2ecc71;
        }
        .badge-warn {
            background: rgba(231,76,60,0.15);
            color: #e74c3c;
            border: 1.5px solid #e74c3c;
        }

        /* Probability Bars */
        .prob-container {
            max-width: 500px;
            margin: 25px auto 0;
            text-align: left;
        }
        .prob-row { margin: 12px 0; }
        .prob-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 14px;
            font-weight: 500;
        }
        .prob-bar {
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            height: 28px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .prob-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 1.2s ease-out;
            display: flex;
            align-items: center;
            padding-left: 10px;
            font-size: 12px;
            font-weight: 600;
        }

        /* Recommendation */
        .recommendation {
            margin-top: 15px;
            padding: 12px 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            font-size: 14px;
            color: #b0b0b0;
            border-left: 3px solid #00d4ff;
        }

        /* Uncertainty Card */
        .uncertainty-info {
            margin-top: 15px;
            font-size: 14px;
        }
        .uncertainty-info p {
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .uncertainty-info span {
            color: #00d4ff;
            font-weight: 600;
        }

        /* Disclaimer */
        .disclaimer {
            background: rgba(231,76,60,0.08);
            border: 1px solid rgba(231,76,60,0.25);
            border-radius: 10px;
            padding: 18px;
            margin-top: 30px;
            text-align: center;
            font-size: 13px;
            color: #b0b0b0;
        }
        .disclaimer strong { color: #e74c3c; }

        /* Error */
        .error-msg {
            display: none;
            background: rgba(231,76,60,0.1);
            border: 1px solid rgba(231,76,60,0.3);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
            color: #e74c3c;
        }

        /* New Analysis Button */
        .new-analysis-btn {
            display: inline-block;
            margin-top: 25px;
            background: rgba(255,255,255,0.1);
            color: #00d4ff;
            border: 1px solid #00d4ff;
            padding: 12px 30px;
            font-size: 14px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .new-analysis-btn:hover {
            background: rgba(0,212,255,0.1);
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            color: #555;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>&#x1FAC1; Explainable AI for Lung Cancer Detection</h1>
        <p>Upload a CT scan image for AI-powered analysis with visual explanations</p>
    </div>

    <div class="container">

        <!-- Upload Section -->
        <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
            <h2>&#x1F4E4; Upload CT Scan Image</h2>
            <p>Supported formats: PNG, JPG, JPEG, BMP (Max 16MB)</p>
            <input type="file" id="fileInput" accept="image/*"
                   style="display:none" onchange="uploadFile(this)">
            <button class="btn" onclick="event.stopPropagation(); document.getElementById('fileInput').click()">
                Choose Image
            </button>
            <br>
            <img id="preview" src="" alt="Preview">
        </div>

        <!-- Error Message -->
        <div class="error-msg" id="errorMsg">
            <p id="errorText"></p>
        </div>

        <!-- Loading -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p><strong>Analyzing CT scan with AI...</strong></p>
            <p style="margin-top:8px; font-size:13px;">
                Generating Grad-CAM heatmap & uncertainty analysis
            </p>
        </div>

        <!-- Results -->
        <div class="results" id="results">
            <div class="grid">

                <!-- Diagnosis Card (Full Width) -->
                <div class="card diagnosis-card">
                    <h3>&#x1F52C; AI Diagnosis Result</h3>

                    <div class="diagnosis-text" id="diagnosis">-</div>

                    <div class="confidence-text">
                        Confidence: <strong id="confidence">-</strong>
                    </div>

                    <div id="badge"></div>

                    <!-- Probability Bars -->
                    <div class="prob-container">
                        <div class="prob-row">
                            <div class="prob-label">
                                <span>&#x2705; Normal</span>
                                <span id="pNormal">0%</span>
                            </div>
                            <div class="prob-bar">
                                <div class="prob-fill" id="bNormal"
                                     style="width:0%; background: linear-gradient(90deg, #27ae60, #2ecc71);">
                                </div>
                            </div>
                        </div>
                        <div class="prob-row">
                            <div class="prob-label">
                                <span>&#x26A0;&#xFE0F; Benign</span>
                                <span id="pBenign">0%</span>
                            </div>
                            <div class="prob-bar">
                                <div class="prob-fill" id="bBenign"
                                     style="width:0%; background: linear-gradient(90deg, #e67e22, #f39c12);">
                                </div>
                            </div>
                        </div>
                        <div class="prob-row">
                            <div class="prob-label">
                                <span>&#x274C; Malignant</span>
                                <span id="pMalignant">0%</span>
                            </div>
                            <div class="prob-bar">
                                <div class="prob-fill" id="bMalignant"
                                     style="width:0%; background: linear-gradient(90deg, #c0392b, #e74c3c);">
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Recommendation -->
                    <div class="recommendation" id="recommendation"></div>

                    <!-- New Analysis Button -->
                    <button class="new-analysis-btn" onclick="resetPage()">
                        &#x1F504; New Analysis
                    </button>
                </div>

                <!-- Original Image -->
                <div class="card">
                    <h3>&#x1F4F7; Original CT Scan</h3>
                    <p class="desc">Uploaded image resized to 224x224</p>
                    <img id="imgOriginal" src="" alt="Original CT Scan">
                </div>

                <!-- Grad-CAM -->
                <div class="card">
                    <h3>&#x1F525; Grad-CAM Explanation</h3>
                    <p class="desc">
                        Red/warm areas = regions the AI focused on most.
                        Blue/cool areas = less important regions.
                        This helps verify the AI looked at the right area.
                    </p>
                    <img id="imgGradcam" src="" alt="Grad-CAM Heatmap">
                </div>

                <!-- Uncertainty Analysis -->
                <div class="card" style="grid-column: 1 / -1;">
                    <h3>&#x1F4CA; Uncertainty & Reliability Analysis</h3>
                    <p class="desc">
                        Monte Carlo Dropout: The model ran 20 predictions with random dropout.
                        Consistent results = reliable prediction. High variance = uncertain.
                    </p>
                    <div class="uncertainty-info" id="uncertaintyInfo">
                        <p>
                            <span style="color:#b0b0b0;">Uncertainty Score:</span>
                            <span id="uncScore">-</span>
                        </p>
                        <p>
                            <span style="color:#b0b0b0;">Predictive Entropy:</span>
                            <span id="uncEntropy">-</span>
                        </p>
                        <p>
                            <span style="color:#b0b0b0;">Risk Level:</span>
                            <span id="riskLevel">-</span>
                        </p>
                        <p>
                            <span style="color:#b0b0b0;">Reliability:</span>
                            <span id="reliability">-</span>
                        </p>
                    </div>
                </div>

            </div>

            <!-- Disclaimer -->
            <div class="disclaimer">
                &#x26A0;&#xFE0F; <strong>Medical Disclaimer:</strong>
                This AI tool is for research and educational purposes only.
                It is NOT a certified medical device and should NOT replace professional
                medical diagnosis. Always consult qualified healthcare professionals
                for medical decisions.
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Explainable & Robust AI Framework for Early Lung Cancer Detection | Final Year Project</p>
        </div>
    </div>

    <!-- JavaScript -->
    <script>
        function uploadFile(input) {
            const file = input.files[0];
            if (!file) return;

            // Hide previous results and errors
            document.getElementById('results').style.display = 'none';
            document.getElementById('errorMsg').style.display = 'none';

            // Show preview
            const preview = document.getElementById('preview');
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';

            // Validate file size
            if (file.size > 16 * 1024 * 1024) {
                showError('File is too large. Maximum size is 16MB.');
                return;
            }

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('uploadArea').style.opacity = '0.5';
            document.getElementById('uploadArea').style.pointerEvents = 'none';

            // Create form data
            const formData = new FormData();
            formData.append('file', file);

            // Send to server
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Server error');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                document.getElementById('uploadArea').style.opacity = '1';
                document.getElementById('uploadArea').style.pointerEvents = 'auto';

                if (data.error) {
                    showError(data.error);
                    return;
                }

                // Show results
                displayResults(data);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('uploadArea').style.opacity = '1';
                document.getElementById('uploadArea').style.pointerEvents = 'auto';
                showError('Error: ' + error.message);
            });
        }

        function displayResults(data) {
            document.getElementById('results').style.display = 'block';

            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

            // === DIAGNOSIS ===
            const diagEl = document.getElementById('diagnosis');
            diagEl.textContent = data.prediction;

            // Remove old classes, add new
            diagEl.classList.remove('risk-low', 'risk-moderate', 'risk-high');
            if (data.risk_level === 'LOW') diagEl.classList.add('risk-low');
            else if (data.risk_level === 'MODERATE') diagEl.classList.add('risk-moderate');
            else diagEl.classList.add('risk-high');

            // === CONFIDENCE ===
            document.getElementById('confidence').textContent =
                (data.confidence * 100).toFixed(1) + '%';

            // === BADGE ===
            const badgeEl = document.getElementById('badge');
            if (data.is_reliable) {
                badgeEl.innerHTML = '<span class="badge badge-ok">&#x2714; Reliable Prediction</span>';
            } else {
                badgeEl.innerHTML = '<span class="badge badge-warn">&#x26A0; Uncertain - Human Review Needed</span>';
            }

            // === PROBABILITIES ===
            const classes = ['Normal', 'Benign', 'Malignant'];
            classes.forEach(function(cls) {
                const value = data.probabilities[cls] || 0;
                const percent = (value * 100).toFixed(1);

                document.getElementById('p' + cls).textContent = percent + '%';

                // Animate bar
                setTimeout(function() {
                    document.getElementById('b' + cls).style.width = percent + '%';
                }, 200);
            });

            // === RECOMMENDATION ===
            document.getElementById('recommendation').textContent =
                '💡 ' + data.recommendation;

            // === IMAGES ===
            document.getElementById('imgOriginal').src = data.original_url;
            document.getElementById('imgGradcam').src = data.gradcam_url;

            // === UNCERTAINTY ===
            document.getElementById('uncScore').textContent =
                data.uncertainty.toFixed(4);
            document.getElementById('uncEntropy').textContent =
                data.entropy.toFixed(4);
            document.getElementById('riskLevel').textContent = data.risk_level;
            document.getElementById('riskLevel').style.color =
                data.risk_level === 'LOW' ? '#2ecc71' :
                data.risk_level === 'MODERATE' ? '#f39c12' : '#e74c3c';
            document.getElementById('reliability').textContent =
                data.is_reliable ? '✓ Reliable' : '⚠ Uncertain';
            document.getElementById('reliability').style.color =
                data.is_reliable ? '#2ecc71' : '#e74c3c';
        }

        function showError(message) {
            const errorDiv = document.getElementById('errorMsg');
            document.getElementById('errorText').textContent = message;
            errorDiv.style.display = 'block';
        }

        function resetPage() {
            // Hide results
            document.getElementById('results').style.display = 'none';
            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('preview').style.display = 'none';

            // Reset file input
            document.getElementById('fileInput').value = '';

            // Reset probability bars
            ['Normal', 'Benign', 'Malignant'].forEach(function(cls) {
                document.getElementById('b' + cls).style.width = '0%';
                document.getElementById('p' + cls).textContent = '0%';
            });

            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        // Drag and drop support
        const uploadArea = document.getElementById('uploadArea');

        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#00d4ff';
            this.style.background = 'rgba(0,212,255,0.1)';
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = 'rgba(0,212,255,0.3)';
            this.style.background = 'rgba(255,255,255,0.05)';
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = 'rgba(0,212,255,0.3)';
            this.style.background = 'rgba(255,255,255,0.05)';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('fileInput').files = files;
                uploadFile(document.getElementById('fileInput'));
            }
        });
    </script>
</body>
</html>'''

    template_path = os.path.join('templates', 'index.html')
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  ✓ Template created: {template_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("STEP 8: WEB APPLICATION")
    print("=" * 60)

    # Create HTML template
    create_template()

    print(f"\n  ✅ Server starting...")
    print(f"  ✅ Open your browser and go to:")
    print(f"")
    print(f"     👉  http://localhost:5000")
    print(f"")
    print(f"  ✅ Upload a CT scan image to get AI analysis")
    print(f"  ✅ Press Ctrl+C to stop the server")
    print(f"\n" + "=" * 60 + "\n")

    app.run(
        debug=False,       # Set to False to avoid model loading twice
        host='0.0.0.0',    # Accessible from any device on network
        port=5000,
        threaded=True
    )
