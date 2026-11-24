# Research Audit Checkpoints

Due to GitHub's file size limits and best practices for version control of large deep learning binaries (>50 MB), the `.pth` model weights stored here are excluded from Git tracking via `.gitignore`.

### Retrained Model Checkpoints
- `retrained_best_model.pth`: 100,934,975 bytes (~100.9 MB). Trained on certified clean splits, saved at epoch 18 with validation loss 0.0417.
- `retrained_final_model.pth`: 100,938,323 bytes (~100.9 MB). Final weights after 20 epochs.

### Training History & Metrics
- `training_history.json`: Complete per-epoch logs (losses, accuracies, learning rates, runtimes).
- `training_history.png`: Training and validation loss/accuracy progression curves.
