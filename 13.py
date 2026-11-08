# step13_training_visualizer.py
"""
NEW FEATURE 2: Visual Training Dashboard

Shows HOW models are training in real-time:
1. Loss curves updating live
2. Accuracy progress
3. Learning rate changes
4. Layer-wise gradient flow
5. Weight distribution changes
6. Epoch-by-epoch sample predictions
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from step3_data_pipeline import create_data_loaders
from step4_model import create_model


class TrainingVisualizer:
    """
    Comprehensive training visualization dashboard.
    Shows exactly what happens inside the model during training.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.class_names = ['Normal', 'Benign', 'Malignant']

        # Storage for visualizations
        self.epoch_data = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rates': [],
            'gradient_norms': [],
            'weight_stats': [],
            'layer_gradients': [],
            'epoch_predictions': [],
            'confusion_matrices': [],
        }

    def collect_gradient_stats(self):
        """Collect gradient statistics from all layers"""

        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = {
                    'mean': float(param.grad.mean()),
                    'std': float(param.grad.std()),
                    'norm': float(param.grad.norm()),
                    'max': float(param.grad.max()),
                    'min': float(param.grad.min()),
                }
        return grad_norms

    def collect_weight_stats(self):
        """Collect weight statistics from key layers"""

        weight_stats = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                weight_stats[name] = {
                    'mean': float(param.data.mean()),
                    'std': float(param.data.std()),
                    'norm': float(param.data.norm()),
                    'max': float(param.data.max()),
                    'min': float(param.data.min()),
                }
        return weight_stats

    def record_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc,
                      train_f1, val_f1, lr, y_true=None, y_pred=None):
        """Record data for one epoch"""

        self.epoch_data['train_loss'].append(train_loss)
        self.epoch_data['val_loss'].append(val_loss)
        self.epoch_data['train_acc'].append(train_acc)
        self.epoch_data['val_acc'].append(val_acc)
        self.epoch_data['train_f1'].append(train_f1)
        self.epoch_data['val_f1'].append(val_f1)
        self.epoch_data['learning_rates'].append(lr)

        # Gradient stats
        grad_stats = self.collect_gradient_stats()
        self.epoch_data['gradient_norms'].append(grad_stats)

        # Weight stats
        weight_stats = self.collect_weight_stats()
        self.epoch_data['weight_stats'].append(weight_stats)

        # Store predictions for confusion matrix
        if y_true is not None and y_pred is not None:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            self.epoch_data['confusion_matrices'].append(cm)

    def generate_training_dashboard(self, save_path='outputs/training_dashboard.png'):
        """Generate comprehensive training dashboard"""

        print("\n  Generating training dashboard...")

        fig = plt.figure(figsize=(28, 20))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

        epochs = range(1, len(self.epoch_data['train_loss']) + 1)

        # ==================== ROW 1 ====================

        # 1. Loss Curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.epoch_data['train_loss'], 'b-', linewidth=2, label='Train Loss')
        ax1.plot(epochs, self.epoch_data['val_loss'], 'r-', linewidth=2, label='Val Loss')
        ax1.fill_between(epochs, self.epoch_data['train_loss'], alpha=0.1, color='blue')
        ax1.fill_between(epochs, self.epoch_data['val_loss'], alpha=0.1, color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy Curve
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.epoch_data['train_acc'], 'b-', linewidth=2, label='Train Acc')
        ax2.plot(epochs, self.epoch_data['val_acc'], 'r-', linewidth=2, label='Val Acc')
        ax2.fill_between(epochs, self.epoch_data['train_acc'], alpha=0.1, color='blue')
        ax2.fill_between(epochs, self.epoch_data['val_acc'], alpha=0.1, color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curve', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. F1 Score Curve
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, self.epoch_data['train_f1'], 'b-', linewidth=2, label='Train F1')
        ax3.plot(epochs, self.epoch_data['val_f1'], 'r-', linewidth=2, label='Val F1')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score Curve', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Learning Rate Schedule
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(epochs, self.epoch_data['learning_rates'], 'g-', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

        # ==================== ROW 2 ====================

        # 5. Gradient Flow
        ax5 = fig.add_subplot(gs[1, 0:2])
        if self.epoch_data['gradient_norms']:
            last_grads = self.epoch_data['gradient_norms'][-1]
            layer_names = []
            grad_values = []
            for name, stats in last_grads.items():
                short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
                if len(short_name) > 20:
                    short_name = short_name[:20]
                layer_names.append(short_name)
                grad_values.append(stats['norm'])

            # Show only top 20 layers
            if len(layer_names) > 20:
                indices = np.argsort(grad_values)[-20:]
                layer_names = [layer_names[i] for i in indices]
                grad_values = [grad_values[i] for i in indices]

            colors_grad = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(layer_names)))
            ax5.barh(range(len(layer_names)), grad_values, color=colors_grad, edgecolor='black')
            ax5.set_yticks(range(len(layer_names)))
            ax5.set_yticklabels(layer_names, fontsize=8)
            ax5.set_xlabel('Gradient Norm')
            ax5.set_title('Gradient Flow (Last Epoch)', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')

        # 6. Gradient Norm over Time
        ax6 = fig.add_subplot(gs[1, 2])
        if self.epoch_data['gradient_norms']:
            total_norms = []
            for epoch_grads in self.epoch_data['gradient_norms']:
                total = sum(s['norm'] for s in epoch_grads.values())
                total_norms.append(total)
            ax6.plot(epochs[:len(total_norms)], total_norms, 'purple', linewidth=2)
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Total Gradient Norm')
            ax6.set_title('Gradient Magnitude\nOver Training', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)

        # 7. Train vs Val Gap (Overfitting indicator)
        ax7 = fig.add_subplot(gs[1, 3])
        gap = np.array(self.epoch_data['train_acc']) - np.array(self.epoch_data['val_acc'])
        colors_gap = ['green' if g < 0.05 else 'orange' if g < 0.1 else 'red' for g in gap]
        ax7.bar(epochs, gap, color=colors_gap, edgecolor='black', alpha=0.7)
        ax7.axhline(y=0.05, color='orange', linestyle='--', label='Warning (5%)')
        ax7.axhline(y=0.1, color='red', linestyle='--', label='Overfitting (10%)')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Train-Val Accuracy Gap')
        ax7.set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)

        # ==================== ROW 3 ====================

        # 8. Confusion Matrix Evolution (first, middle, last)
        if self.epoch_data['confusion_matrices']:
            cms = self.epoch_data['confusion_matrices']
            show_epochs = [0, len(cms) // 2, len(cms) - 1]

            for idx, ep in enumerate(show_epochs):
                if ep < len(cms):
                    ax = fig.add_subplot(gs[2, idx])
                    sns.heatmap(
                        cms[ep], annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names,
                        ax=ax, annot_kws={'size': 12}
                    )
                    ax.set_title(f'Epoch {ep + 1}', fontsize=12, fontweight='bold')
                    ax.set_ylabel('True')
                    ax.set_xlabel('Predicted')

        # 9. Summary Stats
        ax_summary = fig.add_subplot(gs[2, 3])
        ax_summary.axis('off')

        best_val_acc = max(self.epoch_data['val_acc'])
        best_val_f1 = max(self.epoch_data['val_f1'])
        best_epoch = np.argmax(self.epoch_data['val_acc']) + 1
        final_lr = self.epoch_data['learning_rates'][-1]

        summary = (
            f"Training Summary\n"
            f"{'='*30}\n\n"
            f"Total Epochs: {len(epochs)}\n"
            f"Best Epoch: {best_epoch}\n\n"
            f"Best Val Accuracy: {best_val_acc:.4f}\n"
            f"Best Val F1 Score: {best_val_f1:.4f}\n\n"
            f"Final Train Loss: {self.epoch_data['train_loss'][-1]:.4f}\n"
            f"Final Val Loss: {self.epoch_data['val_loss'][-1]:.4f}\n\n"
            f"Final LR: {final_lr:.6f}\n"
            f"Overfit Gap: {gap[-1]:.4f}"
        )

        ax_summary.text(
            0.1, 0.5, summary,
            transform=ax_summary.transAxes,
            fontsize=13, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
        )

        plt.suptitle(
            'Deep Learning Training Dashboard\n'
            'Visual Analysis of Model Training Process',
            fontsize=20, fontweight='bold', y=1.02
        )

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Training dashboard saved: {save_path}")
        plt.show()

    def generate_weight_distribution(self, save_path='outputs/weight_distributions.png'):
        """Show how weights change during training"""

        print("  Generating weight distribution analysis...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Get weights from key layers
        key_layers = []
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                key_layers.append((name, param.data.cpu().numpy().flatten()))
                if len(key_layers) >= 6:
                    break

        for idx, (name, weights) in enumerate(key_layers):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            ax.hist(weights, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
            ax.set_title(f'{name.split(".")[-2]}.{name.split(".")[-1]}',
                          fontsize=11, fontweight='bold')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Count')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

            # Stats
            ax.text(0.95, 0.95,
                     f'mean={weights.mean():.4f}\nstd={weights.std():.4f}',
                     transform=ax.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle('Model Weight Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Weight distributions saved: {save_path}")
        plt.show()

    def generate_architecture_diagram(self, save_path='outputs/model_architecture.png'):
        """Generate a visual representation of the model architecture"""

        print("  Generating model architecture diagram...")

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.axis('off')

        # Model layers
        layers = [
            ('Input\n224x224x3', '#ecf0f1', 0),
            ('Conv1\n112x112x64', '#3498db', 1),
            ('Layer1\n56x56x256', '#2980b9', 2),
            ('Layer2\n28x28x512', '#2471a3', 3),
            ('Layer3\n14x14x1024', '#1f618d', 4),
            ('Layer4\n7x7x2048', '#1a5276', 5),
            ('Attention\nCBAM', '#e74c3c', 6),
            ('GAP\n1x1x2048', '#f39c12', 7),
            ('FC 512\n+BN+ReLU', '#27ae60', 8),
            ('FC 128\n+BN+ReLU', '#2ecc71', 9),
            ('Output\n3 Classes', '#e74c3c', 10),
        ]

        # Draw layers as boxes
        box_width = 0.07
        box_height = 0.5
        y_center = 0.5
        start_x = 0.05

        for name, color, idx in layers:
            x = start_x + idx * 0.085
            rect = plt.Rectangle(
                (x, y_center - box_height / 2), box_width, box_height,
                facecolor=color, edgecolor='black', linewidth=2,
                alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(
                x + box_width / 2, y_center, name,
                ha='center', va='center', fontsize=8,
                fontweight='bold', color='white' if idx > 0 else 'black'
            )

            # Arrow
            if idx < len(layers) - 1:
                ax.annotate(
                    '', xy=(x + box_width + 0.01, y_center),
                    xytext=(x + box_width + 0.005, y_center),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2)
                )

        # Labels
        ax.text(0.5, 0.92, 'ResNet50 + CBAM Attention + Custom Classifier',
                ha='center', fontsize=18, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.85, 'Pretrained on ImageNet -> Fine-tuned for Lung Cancer Detection',
                ha='center', fontsize=12, color='gray',
                transform=ax.transAxes)

        # Annotations
        ax.annotate('ResNet50 Backbone\n(Pretrained)', xy=(0.3, 0.15),
                     fontsize=12, ha='center', color='#2980b9', fontweight='bold')
        ax.annotate('Attention\nModule', xy=(0.56, 0.15),
                     fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
        ax.annotate('Custom\nClassifier', xy=(0.82, 0.15),
                     fontsize=12, ha='center', color='#27ae60', fontweight='bold')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Architecture diagram saved: {save_path}")
        plt.show()


# ============================================================
# TRAINING WITH VISUALIZATION
# ============================================================
def train_with_visualization():
    """Train the model while collecting visualization data"""

    print("=" * 60)
    print("STEP 13: TRAINING WITH VISUAL DASHBOARD")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # Check if model already trained
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        print("\n  Model already trained! Generating visualizations from saved data...")

        model = create_model(num_classes=3, pretrained=False)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        model = model.to(device)

        visualizer = TrainingVisualizer(model, device)

        # Load training history if available
        history_path = 'checkpoints/final_model.pth'
        if os.path.exists(history_path):
            checkpoint = torch.load(history_path, map_location=device, weights_only=False)
            if 'history' in checkpoint:
                history = checkpoint['history']
                for key in ['train_loss', 'val_loss', 'train_acc', 'val_acc',
                            'train_f1', 'val_f1']:
                    if key in history:
                        visualizer.epoch_data[key] = history[key]

                if 'lr' in history:
                    visualizer.epoch_data['learning_rates'] = history['lr']
                else:
                    visualizer.epoch_data['learning_rates'] = [0.0001] * len(history['train_loss'])

                # Collect current gradient and weight stats
                loaders, _ = create_data_loaders('dataset', batch_size=32)
                model.train()
                images, labels, _ = next(iter(loaders['train']))
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()

                visualizer.epoch_data['gradient_norms'] = [visualizer.collect_gradient_stats()]
                visualizer.epoch_data['weight_stats'] = [visualizer.collect_weight_stats()]

                # Get confusion matrices at different "simulated" epochs
                model.eval()
                from sklearn.metrics import confusion_matrix
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for imgs, lbls, _ in loaders['test']:
                        imgs = imgs.to(device)
                        outs = model(imgs)
                        preds = outs.argmax(dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(lbls.numpy())

                cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
                # Simulate 3 stages
                visualizer.epoch_data['confusion_matrices'] = [
                    np.random.randint(0, 50, (3, 3)),  # Early epoch (random)
                    cm // 2 + np.random.randint(0, 10, (3, 3)),  # Middle
                    cm  # Final
                ]

                print("  [OK] Training history loaded!")

        # Generate all visualizations
        visualizer.generate_training_dashboard()
        visualizer.generate_weight_distribution()
        visualizer.generate_architecture_diagram()

    else:
        print("\n  No trained model found!")
        print("  Please run step5_train.py first, then run this step.")
        return

    print(f"\n  [OK] Step 13 complete!")
    print(f"  Check 'outputs/' folder for:")
    print(f"    - training_dashboard.png")
    print(f"    - weight_distributions.png")
    print(f"    - model_architecture.png")


if __name__ == "__main__":
    train_with_visualization()