# step14_enhanced_webapp.py
"""
NEW FEATURE 3: Enhanced Web Application

Adds:
1. Symptom-based risk assessment form
2. Combined CT scan + Symptom analysis
3. Training visualization page
4. Model comparison dashboard
"""

import os
import uuid
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from flask import Flask, render_template_string, request, jsonify
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# ============================================================
# LOAD MODELS
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}")

# CT Scan Model
ct_model = create_model(num_classes=3, pretrained=False)
ct_checkpoint = 'checkpoints/best_model.pth'
if os.path.exists(ct_checkpoint):
    ct_model.load_state_dict(torch.load(ct_checkpoint, map_location=device, weights_only=True))
    print("  [OK] CT scan model loaded")
else:
    print("  [WARN] CT model not found!")
ct_model = ct_model.to(device)
ct_model.eval()

# Symptom Model
symptom_model = None
symptom_scaler = None
symptom_features = None

if os.path.exists('checkpoints/best_symptom_model.joblib'):
    symptom_model = joblib.load('checkpoints/best_symptom_model.joblib')
    print("  [OK] Symptom model loaded")

if os.path.exists('checkpoints/symptom_scaler.joblib'):
    symptom_scaler = joblib.load('checkpoints/symptom_scaler.joblib')

if os.path.exists('checkpoints/symptom_features.joblib'):
    symptom_features = joblib.load('checkpoints/symptom_features.joblib')

# Constants
CT_CLASSES = ['Normal', 'Benign', 'Malignant']
SYMPTOM_CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


# ============================================================
# CT SCAN PROCESSING
# ============================================================
def process_ct_scan(filepath):
    """Process CT scan image"""

    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_display = img.astype(np.float32) / 255.0

    img_normalized = (img_display - MEAN) / STD
    img_tensor = torch.from_numpy(
        img_normalized.transpose(2, 0, 1)
    ).float().unsqueeze(0).to(device)

    ct_model.eval()
    with torch.no_grad():
        output = ct_model(img_tensor)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    # Grad-CAM
    target_layer = ct_model.backbone.layer4[-1]
    cam = GradCAM(model=ct_model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
    overlay = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)
    del cam

    # MC Dropout
    def enable_dropout(m):
        for module in m.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    ct_model.eval()
    enable_dropout(ct_model)
    preds_list = []
    for _ in range(15):
        with torch.no_grad():
            out = ct_model(img_tensor)
            p = F.softmax(out, dim=1)[0].cpu().numpy()
            preds_list.append(p)
    ct_model.eval()

    preds_arr = np.array(preds_list)
    uncertainty = float(preds_arr.std(axis=0)[pred_class])
    is_reliable = (uncertainty < 0.15) and (confidence > 0.7)

    return {
        'prediction': CT_CLASSES[pred_class],
        'confidence': confidence,
        'probabilities': {CT_CLASSES[i]: float(probs[i]) for i in range(3)},
        'uncertainty': uncertainty,
        'is_reliable': is_reliable,
        'overlay': overlay
    }


# ============================================================
# SYMPTOM PROCESSING
# ============================================================
def process_symptoms(symptom_data):
    """Process symptom form data"""

    if symptom_model is None:
        return {'error': 'Symptom model not loaded. Run step12_symptoms.py first.'}

    # Create feature vector in correct order
    features = []
    for feat_name in symptom_features:
        value = symptom_data.get(feat_name, 0)
        features.append(float(value))

    features = np.array(features).reshape(1, -1)

    # Scale
    if symptom_scaler is not None:
        features = symptom_scaler.transform(features)

    # Predict
    pred = symptom_model.predict(features)[0]
    proba = symptom_model.predict_proba(features)[0]

    risk_level = SYMPTOM_CLASSES[pred] if pred < len(SYMPTOM_CLASSES) else f"Class {pred}"

    recommendations = {
        'Low Risk': 'Your symptoms indicate low risk. Continue healthy lifestyle and regular check-ups.',
        'Medium Risk': 'Your symptoms indicate moderate risk. Consider consulting a pulmonologist for screening.',
        'High Risk': 'Your symptoms indicate high risk. Strongly recommend immediate medical consultation and CT scan screening.'
    }

    return {
        'risk_level': risk_level,
        'probabilities': {SYMPTOM_CLASSES[i]: float(proba[i]) for i in range(len(proba))},
        'recommendation': recommendations.get(risk_level, 'Please consult a doctor.'),
        'risk_score': float(proba[pred])
    }


# ============================================================
# HTML TEMPLATE
# ============================================================
HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Cancer Detection - Enhanced AI System</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'Segoe UI',Arial,sans-serif;background:linear-gradient(135deg,#0c0c1d,#1a1a3e);color:#e0e0e0;min-height:100vh}
        .header{background:rgba(255,255,255,0.05);padding:20px;text-align:center;border-bottom:1px solid rgba(255,255,255,0.1)}
        .header h1{color:#00d4ff;font-size:24px}
        .header p{color:#8892b0;font-size:13px;margin-top:5px}
        .container{max-width:1400px;margin:20px auto;padding:0 15px}
        .tabs{display:flex;gap:5px;margin-bottom:20px;flex-wrap:wrap}
        .tab{padding:12px 25px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:8px;cursor:pointer;color:#8892b0;font-size:14px;transition:all 0.3s}
        .tab:hover,.tab.active{background:rgba(0,212,255,0.1);color:#00d4ff;border-color:#00d4ff}
        .tab-content{display:none}
        .tab-content.active{display:block}
        .card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:25px;margin-bottom:15px}
        .card h3{color:#00d4ff;margin-bottom:15px;font-size:16px}
        .card img{width:100%;border-radius:8px;margin-top:10px}
        .btn{background:linear-gradient(135deg,#00d4ff,#0077b6);color:#fff;border:none;padding:12px 30px;font-size:15px;border-radius:8px;cursor:pointer;transition:all 0.3s}
        .btn:hover{transform:translateY(-2px);box-shadow:0 5px 15px rgba(0,212,255,0.3)}
        .form-group{margin-bottom:15px}
        .form-group label{display:block;color:#8892b0;margin-bottom:5px;font-size:13px}
        .form-group select,.form-group input{width:100%;padding:10px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:6px;color:#e0e0e0;font-size:14px}
        .form-group select option{background:#1a1a3e;color:#e0e0e0}
        .grid2{display:grid;grid-template-columns:1fr 1fr;gap:15px}
        .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px}
        @media(max-width:768px){.grid2,.grid3{grid-template-columns:1fr}}
        .result-box{text-align:center;padding:25px}
        .result-text{font-size:32px;font-weight:800;margin:10px 0}
        .risk-low,.risk-LOW{color:#2ecc71}
        .risk-medium,.risk-MODERATE{color:#f39c12}
        .risk-high,.risk-HIGH{color:#e74c3c}
        .prob-bar{background:rgba(255,255,255,0.08);border-radius:8px;height:25px;overflow:hidden;margin:5px 0}
        .prob-fill{height:100%;border-radius:8px;transition:width 1s}
        .prob-label{display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px}
        .badge{display:inline-block;padding:5px 15px;border-radius:20px;font-size:12px;font-weight:700}
        .badge-ok{background:rgba(46,204,113,0.15);color:#2ecc71;border:1px solid #2ecc71}
        .badge-warn{background:rgba(231,76,60,0.15);color:#e74c3c;border:1px solid #e74c3c}
        .loading{display:none;text-align:center;padding:30px}
        .spinner{border:4px solid rgba(255,255,255,0.1);border-top:4px solid #00d4ff;border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:0 auto 15px}
        @keyframes spin{to{transform:rotate(360deg)}}
        .upload-area{border:2px dashed rgba(0,212,255,0.3);border-radius:12px;padding:40px;text-align:center;cursor:pointer;transition:all 0.3s}
        .upload-area:hover{border-color:#00d4ff;background:rgba(0,212,255,0.05)}
        #preview{max-width:200px;margin-top:10px;border-radius:8px;display:none}
        .disclaimer{background:rgba(231,76,60,0.08);border:1px solid rgba(231,76,60,0.2);border-radius:8px;padding:12px;margin-top:15px;text-align:center;font-size:12px}
        .section-title{color:#00d4ff;font-size:20px;font-weight:bold;margin-bottom:15px;padding-bottom:8px;border-bottom:2px solid rgba(0,212,255,0.3)}
    </style>
</head>
<body>
    <div class="header">
        <h1>Explainable & Robust AI for Lung Cancer Detection</h1>
        <p>CT Scan Analysis | Symptom Assessment | Model Comparison | Training Visualization</p>
    </div>

    <div class="container">
        <div class="tabs">
            <div class="tab active" onclick="showTab('ct')">CT Scan Analysis</div>
            <div class="tab" onclick="showTab('symptoms')">Symptom Assessment</div>
            <div class="tab" onclick="showTab('training')">Training Visualization</div>
            <div class="tab" onclick="showTab('comparison')">Model Comparison</div>
        </div>

        <!-- TAB 1: CT SCAN -->
        <div id="tab-ct" class="tab-content active">
            <div class="section-title">CT Scan Analysis with Explainable AI</div>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <h3>Upload CT Scan Image</h3>
                <p style="color:#8892b0;margin:10px 0">PNG, JPG, JPEG (Max 16MB)</p>
                <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="uploadCT(this)">
                <button class="btn" onclick="event.stopPropagation()">Choose Image</button>
                <br><img id="preview" src="">
            </div>
            <div class="loading" id="ctLoading"><div class="spinner"></div><p>Analyzing CT scan...</p></div>
            <div id="ctResults" style="display:none"></div>
        </div>

        <!-- TAB 2: SYMPTOMS -->
        <div id="tab-symptoms" class="tab-content">
            <div class="section-title">Symptom-Based Risk Assessment</div>
            <div class="card">
                <h3>Enter Patient Information & Symptoms</h3>
                <form id="symptomForm" onsubmit="submitSymptoms(event)">
                    <div class="grid3">
                        <div class="form-group">
                            <label>Age</label>
                            <input type="number" name="Age" value="55" min="20" max="90">
                        </div>
                        <div class="form-group">
                            <label>Gender</label>
                            <select name="Gender"><option value="0">Female</option><option value="1">Male</option></select>
                        </div>
                        <div class="form-group">
                            <label>Smoking Status</label>
                            <select name="Smoking"><option value="0">Never</option><option value="1">Former</option><option value="2">Current</option></select>
                        </div>
                        <div class="form-group">
                            <label>Smoking Years</label>
                            <input type="number" name="Smoking_Years" value="0" min="0" max="50">
                        </div>
                        <div class="form-group">
                            <label>Alcohol Use</label>
                            <select name="Alcohol_Use"><option value="0">None</option><option value="1">Moderate</option><option value="2">Heavy</option></select>
                        </div>
                        <div class="form-group">
                            <label>Obesity</label>
                            <select name="Obesity"><option value="0">No</option><option value="1">Yes</option></select>
                        </div>
                    </div>
                    <h3 style="margin-top:15px">Symptoms</h3>
                    <div class="grid3">
                        <div class="form-group"><label>Persistent Cough</label><select name="Persistent_Cough"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Coughing Blood</label><select name="Coughing_Blood"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Shortness of Breath</label><select name="Shortness_of_Breath"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Chest Pain</label><select name="Chest_Pain"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Wheezing</label><select name="Wheezing"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Hoarseness</label><select name="Hoarseness"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Fatigue</label><select name="Fatigue"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Weight Loss</label><select name="Weight_Loss"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Loss of Appetite</label><select name="Loss_of_Appetite"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Difficulty Swallowing</label><select name="Difficulty_Swallowing"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Finger Clubbing</label><select name="Finger_Clubbing"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Frequent Cold</label><select name="Frequent_Cold"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Dry Cough</label><select name="Dry_Cough"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Snoring</label><select name="Snoring"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Swallowing Difficulty</label><select name="Swallowing_Difficulty"><option value="0">No</option><option value="1">Yes</option></select></div>
                    </div>
                    <h3 style="margin-top:15px">Medical History</h3>
                    <div class="grid3">
                        <div class="form-group"><label>Air Pollution Exposure</label><select name="Air_Pollution_Exposure"><option value="0">Low</option><option value="1">Medium</option><option value="2">High</option></select></div>
                        <div class="form-group"><label>Occupational Hazards</label><select name="Occupational_Hazards"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Dust Allergy</label><select name="Dust_Allergy"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Family History</label><select name="Family_History"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Genetic Risk</label><select name="Genetic_Risk"><option value="0">Low</option><option value="1">Medium</option><option value="2">High</option></select></div>
                        <div class="form-group"><label>Chronic Lung Disease</label><select name="Chronic_Lung_Disease"><option value="0">No</option><option value="1">Yes</option></select></div>
                        <div class="form-group"><label>Previous Cancer</label><select name="Previous_Cancer"><option value="0">No</option><option value="1">Yes</option></select></div>
                    </div>
                    <br>
                    <button type="submit" class="btn">Assess Risk</button>
                </form>
            </div>
            <div class="loading" id="sympLoading"><div class="spinner"></div><p>Analyzing symptoms...</p></div>
            <div id="sympResults" style="display:none"></div>
        </div>

        <!-- TAB 3: TRAINING VIZ -->
        <div id="tab-training" class="tab-content">
            <div class="section-title">Training Process Visualization</div>
            <div class="grid2">
                <div class="card"><h3>Training Dashboard</h3><img src="/static/training_dashboard.png" onerror="this.parentElement.innerHTML='<h3>Training Dashboard</h3><p>Run step13_training_visualizer.py to generate</p>'"></div>
                <div class="card"><h3>Model Architecture</h3><img src="/static/model_architecture.png" onerror="this.parentElement.innerHTML='<h3>Model Architecture</h3><p>Run step13_training_visualizer.py to generate</p>'"></div>
                <div class="card"><h3>Weight Distributions</h3><img src="/static/weight_distributions.png" onerror="this.parentElement.innerHTML='<h3>Weight Distributions</h3><p>Run step13_training_visualizer.py to generate</p>'"></div>
                <div class="card"><h3>Training History</h3><img src="/static/training_history.png" onerror="this.parentElement.innerHTML='<h3>Training History</h3><p>Run step5_train.py to generate</p>'"></div>
            </div>
        </div>

        <!-- TAB 4: MODEL COMPARISON -->
        <div id="tab-comparison" class="tab-content">
            <div class="section-title">ML Model Comparison</div>
            <div class="grid2">
                <div class="card"><h3>Model Performance Comparison</h3><img src="/static/symptom_model_comparison.png" onerror="this.parentElement.innerHTML='<h3>Model Comparison</h3><p>Run step12_symptoms.py to generate</p>'"></div>
                <div class="card"><h3>Feature Importance</h3><img src="/static/symptom_feature_importance.png" onerror="this.parentElement.innerHTML='<h3>Feature Importance</h3><p>Run step12_symptoms.py to generate</p>'"></div>
                <div class="card"><h3>ROC Curves</h3><img src="/static/symptom_roc_curves.png" onerror="this.parentElement.innerHTML='<h3>ROC Curves</h3><p>Run step12_symptoms.py to generate</p>'"></div>
                <div class="card"><h3>Confusion Matrices</h3><img src="/static/symptom_confusion_matrices.png" onerror="this.parentElement.innerHTML='<h3>Confusion Matrices</h3><p>Run step12_symptoms.py to generate</p>'"></div>
            </div>
        </div>

        <div class="disclaimer">This is an AI research tool for educational purposes only. NOT a medical device. Always consult healthcare professionals.</div>
    </div>

    <script>
    function showTab(name){
        document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
        document.getElementById('tab-'+name).classList.add('active');
        event.target.classList.add('active');
    }
    function uploadCT(input){
        if(!input.files[0])return;
        document.getElementById('preview').src=URL.createObjectURL(input.files[0]);
        document.getElementById('preview').style.display='block';
        document.getElementById('ctLoading').style.display='block';
        document.getElementById('ctResults').style.display='none';
        const fd=new FormData();fd.append('file',input.files[0]);
        fetch('/predict_ct',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
            document.getElementById('ctLoading').style.display='none';
            if(d.error){alert(d.error);return}
            let h='<div class="card result-box"><h3>AI Diagnosis</h3>';
            h+='<div class="result-text risk-'+d.risk_level+'">'+d.prediction+'</div>';
            h+='<p>Confidence: <b>'+(d.confidence*100).toFixed(1)+'%</b></p>';
            h+=d.is_reliable?'<span class="badge badge-ok">Reliable</span>':'<span class="badge badge-warn">Uncertain</span>';
            h+='<p style="margin-top:10px;color:#8892b0">'+d.recommendation+'</p></div>';
            h+='<div class="grid2"><div class="card"><h3>Original</h3><img src="'+d.original_url+'"></div>';
            h+='<div class="card"><h3>Grad-CAM</h3><img src="'+d.gradcam_url+'"></div></div>';
            document.getElementById('ctResults').innerHTML=h;
            document.getElementById('ctResults').style.display='block';
        }).catch(e=>{document.getElementById('ctLoading').style.display='none';alert(e)});
    }
    function submitSymptoms(e){
        e.preventDefault();
        document.getElementById('sympLoading').style.display='block';
        document.getElementById('sympResults').style.display='none';
        const fd=new FormData(document.getElementById('symptomForm'));
        const data={};fd.forEach((v,k)=>data[k]=parseFloat(v));
        fetch('/predict_symptoms',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(r=>r.json()).then(d=>{
            document.getElementById('sympLoading').style.display='none';
            if(d.error){alert(d.error);return}
            let cls=d.risk_level.toLowerCase().replace(' ','-');
            let h='<div class="card result-box"><h3>Risk Assessment Result</h3>';
            h+='<div class="result-text risk-'+cls+'">'+d.risk_level+'</div>';
            h+='<p>Risk Score: <b>'+(d.risk_score*100).toFixed(1)+'%</b></p>';
            h+='<p style="margin-top:10px;color:#8892b0">'+d.recommendation+'</p></div>';
            document.getElementById('sympResults').innerHTML=h;
            document.getElementById('sympResults').style.display='block';
        }).catch(e=>{document.getElementById('sympLoading').style.display='none';alert(e)});
    }
    </script>
</body>
</html>'''


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict_ct', methods=['POST'])
def predict_ct():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = process_ct_scan(filepath)

        gcam_name = 'gradcam_' + filename.rsplit('.', 1)[0] + '.png'
        gcam_path = os.path.join(app.config['RESULTS_FOLDER'], gcam_name)
        Image.fromarray(result['overlay']).save(gcam_path)

        risk_map = {'Normal': 'LOW', 'Benign': 'MODERATE', 'Malignant': 'HIGH'}
        rec_map = {
            'Normal': 'No significant abnormalities detected.',
            'Benign': 'Benign nodule detected. Follow-up recommended.',
            'Malignant': 'Suspicious features detected. Immediate consultation recommended.'
        }
        return jsonify({
            'success': True, 'prediction': result['prediction'],
            'confidence': result['confidence'], 'probabilities': result['probabilities'],
            'uncertainty': result['uncertainty'], 'is_reliable': result['is_reliable'],
            'risk_level': risk_map[result['prediction']],
            'recommendation': rec_map[result['prediction']],
            'original_url': f'/static/uploads/{filename}',
            'gradcam_url': f'/static/results/{gcam_name}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    try:
        data = request.get_json()
        result = process_symptoms(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# COPY OUTPUT IMAGES TO STATIC
# ============================================================
def copy_outputs_to_static():
    """Copy output images to static folder for web display"""
    import shutil

    images_to_copy = [
        'training_dashboard.png', 'weight_distributions.png',
        'model_architecture.png', 'training_history.png',
        'symptom_model_comparison.png', 'symptom_feature_importance.png',
        'symptom_roc_curves.png', 'symptom_confusion_matrices.png',
    ]

    for img in images_to_copy:
        src = os.path.join('outputs', img)
        dst = os.path.join('static', img)
        if os.path.exists(src):
            shutil.copy2(src, dst)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ENHANCED WEB APPLICATION")
    print("=" * 60)

    copy_outputs_to_static()

    print(f"\n  Server starting...")
    print(f"  Open: http://localhost:5000")
    print(f"  Press Ctrl+C to stop\n")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)