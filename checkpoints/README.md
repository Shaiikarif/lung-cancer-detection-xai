# Checkpoints Directory

Pretrained and fine-tuned model weights are stored here.
By default, large binary weights (`*.pth`, `*.joblib`) are excluded from Git version control via `.gitignore` to prevent repository bloat.

### Expected Files:
- `best_model.pth` (~100 MB): ResNet50 + CBAM Attention trained weights on IQ-OTH/NCCD dataset.
- `final_model.pth`: Final epoch weights.

To obtain or train model weights, refer to the root `README.md` instructions (`python scripts/train.py`).
