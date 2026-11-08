# verify_setup.py
# Run this to check everything is installed correctly

import sys
print(f"Python Version: {sys.version}")
print("-" * 50)

libraries = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'sklearn': 'Scikit-learn',
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'pytorch_grad_cam': 'Grad-CAM',
    'lime': 'LIME',
    'flask': 'Flask',
    'tqdm': 'TQDM',
    'albumentations': 'Albumentations',
}

all_good = True
for module, name in libraries.items():
    try:
        lib = __import__(module)
        version = getattr(lib, '__version__', 'installed')
        print(f"  [OK] {name:20s} : {version}")
    except ImportError:
        print(f"  [MISSING] {name:20s} : NOT INSTALLED")
        all_good = False

print("-" * 50)

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"  [OK] GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"  [INFO] GPU: Not available (will use CPU - training will be slower)")

print("-" * 50)
if all_good:
    print("\n  [OK] ALL LIBRARIES INSTALLED SUCCESSFULLY!")
    print("  You are ready to proceed to Step 2.")
else:
    print("\n  [FAIL] Some libraries are missing.")
    print("  Run: pip install -r requirements.txt")