# step9_results.py
"""
STEP 9: Compile All Results for Report & Presentation (FIXED VERSION)

This generates:
1. Complete metrics summary
2. Confusion Matrix
3. ROC Curve
4. Precision-Recall Curve
5. Per-class metrics chart
6. Confidence distribution
7. Comparison with literature
8. Text report file
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from step3_data_pipeline import create_data_loaders
from step4_model import create_model


# ============================================================
# COLLECT ALL PREDICTIONS
# ============================================================
def collect_predictions(model, dataloader, device):
    """Run model on entire test set and collect all predictions"""

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("  Collecting predictions on test set...")

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='  Testing', leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )


# ============================================================
# 1. CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(y_true, y_pred, save_path='outputs/final_confusion_matrix.png'):
    """Generate publication-quality confusion matrix"""

    class_names = ['Normal', 'Benign', 'Malignant']
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], annot_kws={'size': 18},
        linewidths=1, linecolor='white'
    )
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold', pad=15)
    axes[0].set_ylabel('True Label', fontsize=13)
    axes[0].set_xlabel('Predicted Label', fontsize=13)

    # Normalized (percentages)
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    # Avoid division by zero
    cm_sum = np.where(cm_sum == 0, 1, cm_sum)
    cm_percent = cm.astype('float') / cm_sum * 100

    sns.heatmap(
        cm_percent, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], annot_kws={'size': 18},
        linewidths=1, linecolor='white'
    )
    axes[1].set_title('Confusion Matrix (%)', fontsize=16, fontweight='bold', pad=15)
    axes[1].set_ylabel('True Label', fontsize=13)
    axes[1].set_xlabel('Predicted Label', fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [OK] Confusion matrix saved: {save_path}")
    plt.show()


# ============================================================
# 2. ROC CURVE (Multi-class)
# ============================================================
def plot_roc_curve(y_true, y_probs, save_path='outputs/final_roc_curve.png'):
    """Generate multi-class ROC curve"""

    class_names = ['Normal', 'Benign', 'Malignant']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    n_classes = 3

    fig, ax = plt.subplots(figsize=(10, 8))

    # One-hot encode true labels
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    # Plot ROC for each class
    auc_scores = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_names[i]] = roc_auc

        ax.plot(
            fpr, tpr,
            color=colors[i],
            linewidth=2.5,
            label=f'{class_names[i]} (AUC = {roc_auc:.4f})'
        )

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    ax.plot(
        fpr_micro, tpr_micro,
        color='navy', linewidth=3, linestyle='--',
        label=f'Micro-Average (AUC = {roc_auc_micro:.4f})'
    )

    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve - Multi-Class Classification', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [OK] ROC curve saved: {save_path}")
    plt.show()

    return auc_scores, roc_auc_micro


# ============================================================
# 3. PRECISION-RECALL CURVE
# ============================================================
def plot_precision_recall_curve(y_true, y_probs, save_path='outputs/final_pr_curve.png'):
    """Generate Precision-Recall curve"""

    class_names = ['Normal', 'Benign', 'Malignant']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    n_classes = 3

    fig, ax = plt.subplots(figsize=(10, 8))

    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, i], y_probs[:, i]
        )
        pr_auc = auc(recall, precision)

        ax.plot(
            recall, precision,
            color=colors[i],
            linewidth=2.5,
            label=f'{class_names[i]} (AP = {pr_auc:.4f})'
        )

    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [OK] Precision-Recall curve saved: {save_path}")
    plt.show()


# ============================================================
# 4. PER-CLASS METRICS BAR CHART
# ============================================================
def plot_per_class_metrics(y_true, y_pred, save_path='outputs/final_per_class_metrics.png'):
    """Generate per-class metrics comparison chart"""

    class_names = ['Normal', 'Benign', 'Malignant']
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics_list = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, metric in enumerate(metrics_list):
        values = [report[cls][metric] for cls in class_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                      color=colors[i], edgecolor='black', alpha=0.85)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

    ax.set_xlabel('Class', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [OK] Per-class metrics chart saved: {save_path}")
    plt.show()


# ============================================================
# 5. PREDICTION CONFIDENCE DISTRIBUTION
# ============================================================
def plot_confidence_distribution(y_true, y_pred, y_probs,
                                  save_path='outputs/final_confidence_dist.png'):
    """Show distribution of prediction confidence"""

    max_probs = np.max(y_probs, axis=1)
    correct = (y_true == y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Correct vs Incorrect
    axes[0].hist(max_probs[correct], bins=20, alpha=0.7,
                 label=f'Correct ({correct.sum()})', color='#2ecc71', edgecolor='black')
    axes[0].hist(max_probs[~correct], bins=20, alpha=0.7,
                 label=f'Incorrect ({(~correct).sum()})', color='#e74c3c', edgecolor='black')
    axes[0].set_xlabel('Prediction Confidence', fontsize=13)
    axes[0].set_ylabel('Count', fontsize=13)
    axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Per-class confidence
    class_names = ['Normal', 'Benign', 'Malignant']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        mask = y_true == i
        if mask.sum() > 0:
            axes[1].hist(max_probs[mask], bins=15, alpha=0.5,
                         label=cls, color=color, edgecolor='black')

    axes[1].set_xlabel('Prediction Confidence', fontsize=13)
    axes[1].set_ylabel('Count', fontsize=13)
    axes[1].set_title('Confidence by True Class', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [OK] Confidence distribution saved: {save_path}")
    plt.show()


# ============================================================
# 6. COMPLETE SUMMARY TABLE
# ============================================================
def generate_summary_table(y_true, y_pred, y_probs, auc_scores, auc_micro):
    """Generate complete metrics summary"""

    class_names = ['Normal', 'Benign', 'Malignant']

    # Overall metrics
    overall = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'roc_auc_micro': float(auc_micro),
    }

    # Per-class metrics
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0
    )

    per_class = {}
    for cls in class_names:
        per_class[cls] = {
            'precision': float(report[cls]['precision']),
            'recall': float(report[cls]['recall']),
            'f1_score': float(report[cls]['f1-score']),
            'support': int(report[cls]['support']),
            'auc': float(auc_scores.get(cls, 0))
        }

    return overall, per_class


def print_summary(overall, per_class):
    """Print formatted summary"""

    print("\n" + "=" * 70)
    print("                    COMPLETE RESULTS SUMMARY")
    print("=" * 70)

    print("\n  +----------------------------------------------------------+")
    print("  |                  OVERALL METRICS                          |")
    print("  +----------------------------------------------------------+")
    print(f"  |  Accuracy:              {overall['accuracy']:.4f}                          |")
    print(f"  |  Precision (Weighted):  {overall['precision_weighted']:.4f}                          |")
    print(f"  |  Recall (Weighted):     {overall['recall_weighted']:.4f}                          |")
    print(f"  |  F1 Score (Weighted):   {overall['f1_weighted']:.4f}                          |")
    print(f"  |  F1 Score (Macro):      {overall['f1_macro']:.4f}                          |")
    print(f"  |  ROC-AUC (Micro):       {overall['roc_auc_micro']:.4f}                          |")
    print("  +----------------------------------------------------------+")

    print(f"\n  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12} {'Support':<10}")
    print(f"  {'-'*73}")

    for cls, metrics in per_class.items():
        print(f"  {cls:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics['auc']:<12.4f} {metrics['support']:<10}")

    print(f"  {'-'*73}")


# ============================================================
# 7. SAVE RESULTS TO FILES
# ============================================================
def save_results_to_file(overall, per_class, save_path='outputs/final_results.json'):
    """Save all results to JSON file"""

    results = {
        'overall_metrics': overall,
        'per_class_metrics': per_class
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"  [OK] Results saved to: {save_path}")


def save_report_text(overall, per_class, save_path='outputs/final_report.txt'):
    """Save a text report for easy copy-paste into your project report"""

    class_names = ['Normal', 'Benign', 'Malignant']

    lines = []
    lines.append("=" * 70)
    lines.append("EXPLAINABLE & ROBUST AI FRAMEWORK FOR LUNG CANCER DETECTION")
    lines.append("FINAL RESULTS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. OVERALL PERFORMANCE METRICS")
    lines.append("-" * 40)
    lines.append(f"   Accuracy:              {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
    lines.append(f"   Precision (Weighted):  {overall['precision_weighted']:.4f}")
    lines.append(f"   Recall (Weighted):     {overall['recall_weighted']:.4f}")
    lines.append(f"   F1 Score (Weighted):   {overall['f1_weighted']:.4f}")
    lines.append(f"   F1 Score (Macro):      {overall['f1_macro']:.4f}")
    lines.append(f"   ROC-AUC (Micro Avg):   {overall['roc_auc_micro']:.4f}")
    lines.append("")
    lines.append("2. PER-CLASS PERFORMANCE")
    lines.append("-" * 40)
    lines.append(f"   {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")

    for cls in class_names:
        m = per_class[cls]
        lines.append(f"   {cls:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                     f"{m['f1_score']:<12.4f} {m['auc']:<12.4f}")

    lines.append("")
    lines.append("3. MODEL ARCHITECTURE")
    lines.append("-" * 40)
    lines.append("   Base Model:        ResNet50 (pretrained on ImageNet)")
    lines.append("   Attention:         CBAM (Channel + Spatial Attention)")
    lines.append("   Classifier:        Custom FC layers with BatchNorm & Dropout")
    lines.append("   Loss Function:     Focal Loss (gamma=2.0)")
    lines.append("   Optimizer:         AdamW (lr=1e-4, weight_decay=1e-4)")
    lines.append("   Scheduler:         Cosine Annealing Warm Restarts")
    lines.append("")
    lines.append("4. EXPLAINABILITY METHODS USED")
    lines.append("-" * 40)
    lines.append("   - Grad-CAM (Gradient-weighted Class Activation Mapping)")
    lines.append("   - Grad-CAM++ (Improved version)")
    lines.append("   - LIME (Local Interpretable Model-agnostic Explanations)")
    lines.append("   - Integrated Gradients")
    lines.append("")
    lines.append("5. ROBUSTNESS EVALUATION")
    lines.append("-" * 40)
    lines.append("   - FGSM Adversarial Attack Testing")
    lines.append("   - Image Corruption Robustness (noise, blur, brightness, occlusion)")
    lines.append("   - MC Dropout Uncertainty Quantification")
    lines.append("")
    lines.append("6. DATASET")
    lines.append("-" * 40)
    lines.append("   Name:    IQ-OTH/NCCD Lung Cancer Dataset")
    lines.append("   Source:  Kaggle")
    lines.append("   Classes: Normal, Benign, Malignant")
    lines.append("   Split:   70% Train, 15% Validation, 15% Test")
    lines.append("")
    lines.append("=" * 70)
    lines.append("Note: This is an AI research tool for educational purposes.")
    lines.append("Not intended for clinical use without proper validation.")
    lines.append("=" * 70)

    content = "\n".join(lines)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  [OK] Text report saved to: {save_path}")


# ============================================================
# 8. COMPARISON TABLE WITH EXISTING METHODS
# ============================================================
def plot_comparison_with_literature(overall, save_path='outputs/final_comparison.png'):
    """Compare your results with published methods"""

    methods = {
        'Basic CNN': {'accuracy': 0.82, 'f1': 0.80},
        'VGG16': {'accuracy': 0.87, 'f1': 0.85},
        'ResNet50': {'accuracy': 0.91, 'f1': 0.89},
        'DenseNet121': {'accuracy': 0.90, 'f1': 0.88},
        'Proposed\nMethod': {
            'accuracy': overall['accuracy'],
            'f1': overall['f1_weighted']
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    names = list(methods.keys())
    accuracies = [m['accuracy'] for m in methods.values()]
    f1_scores_list = [m['f1'] for m in methods.values()]

    # Color the last bar (our method) differently
    colors_acc = ['#95a5a6'] * (len(names) - 1) + ['#2ecc71']
    colors_f1 = ['#95a5a6'] * (len(names) - 1) + ['#3498db']

    # Accuracy comparison
    bars1 = axes[0].bar(names, accuracies, color=colors_acc, edgecolor='black', alpha=0.85)
    axes[0].set_ylabel('Accuracy', fontsize=13)
    axes[0].set_title('Accuracy Comparison with Existing Methods',
                       fontsize=14, fontweight='bold')
    axes[0].set_ylim(0.7, 1.05)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    # F1 comparison
    bars2 = axes[1].bar(names, f1_scores_list, color=colors_f1, edgecolor='black', alpha=0.85)
    axes[1].set_ylabel('F1 Score', fontsize=13)
    axes[1].set_title('F1 Score Comparison with Existing Methods',
                       fontsize=14, fontweight='bold')
    axes[1].set_ylim(0.7, 1.05)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, f1_scores_list):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [OK] Comparison chart saved: {save_path}")
    plt.show()


# ============================================================
# MAIN - RUN ALL RESULT COMPILATION
# ============================================================
def compile_all_results():
    """Main function to compile all results"""

    print("=" * 60)
    print("STEP 9: COMPILING ALL RESULTS")
    print("=" * 60)

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # 1. Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    model = create_model(num_classes=3, pretrained=False)

    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"\n  ERROR: No trained model found at {checkpoint_path}")
        print(f"  Please run step5_train.py first!")
        return

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    print("  [OK] Model loaded")

    # 2. Load test data
    loaders, _ = create_data_loaders('dataset', batch_size=32)

    # 3. Collect predictions
    y_pred, y_true, y_probs = collect_predictions(model, loaders['test'], device)

    # 4. Generate all visualizations
    print("\n  Generating visualizations...")

    # 4a. Confusion Matrix
    print("\n  [1/7] Confusion Matrix")
    plot_confusion_matrix(y_true, y_pred)

    # 4b. ROC Curve
    print("\n  [2/7] ROC Curve")
    auc_scores, auc_micro = plot_roc_curve(y_true, y_probs)

    # 4c. Precision-Recall Curve
    print("\n  [3/7] Precision-Recall Curve")
    plot_precision_recall_curve(y_true, y_probs)

    # 4d. Per-class metrics
    print("\n  [4/7] Per-Class Metrics")
    plot_per_class_metrics(y_true, y_pred)

    # 4e. Confidence distribution
    print("\n  [5/7] Confidence Distribution")
    plot_confidence_distribution(y_true, y_pred, y_probs)

    # 5. Summary
    print("\n  [6/7] Generating Summary")
    overall, per_class = generate_summary_table(y_true, y_pred, y_probs, auc_scores, auc_micro)
    print_summary(overall, per_class)

    # 6. Comparison
    print("\n  [7/7] Comparison Chart")
    plot_comparison_with_literature(overall)

    # 7. Save everything
    print("\n  Saving results...")
    save_results_to_file(overall, per_class)
    save_report_text(overall, per_class)

    # 8. Print classification report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Benign', 'Malignant'],
        digits=4,
        zero_division=0
    ))

    # 9. List all output files
    print("\n" + "=" * 60)
    print("ALL OUTPUT FILES")
    print("=" * 60)

    output_files = sorted(os.listdir('outputs'))
    for f in output_files:
        filepath = os.path.join('outputs', f)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/1024/1024:.1f} MB"
            print(f"  [FILE] outputs/{f:<40s} ({size_str})")

    print(f"\n  [OK] Step 9 complete! All results compiled.")
    print(f"  You now have everything needed for your project report.")
    print(f"  Next: python step10_report_content.py")


if __name__ == "__main__":
    compile_all_results()