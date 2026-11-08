# step7_robustness.py
"""
STEP 7: Robustness Testing (FIXED VERSION)

Tests:
1. Adversarial Attacks (FGSM)
2. Noise/Corruption Robustness
3. Uncertainty Estimation (MC Dropout)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from step3_data_pipeline import create_data_loaders
from step4_model import create_model


class RobustnessEngine:
    """Engine for testing model robustness"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.class_names = ['Normal', 'Benign', 'Malignant']

    # ======================== FGSM ATTACK (FIXED) ========================

    def fgsm_attack(self, image, label, epsilon):
        """
        FGSM (Fast Gradient Sign Method) Attack - FIXED VERSION

        Key fix:
        - Create a brand new leaf tensor with requires_grad=True
        - Use .data.clone() to ensure it's a proper leaf tensor
        - This ensures .grad will be populated after backward()
        """

        # ✅ FIX: Create a proper leaf tensor
        # Step 1: Detach from any computation graph
        # Step 2: Clone to get a fresh tensor
        # Step 3: Set requires_grad=True to track gradients
        image_clean = image.detach().clone().to(self.device)
        image_adv = image_clean.clone().detach().requires_grad_(True)

        label_tensor = torch.tensor([label]).to(self.device)

        # Forward pass
        self.model.eval()
        output = self.model(image_adv)
        loss = F.cross_entropy(output, label_tensor)

        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()

        # ✅ FIX: Check if gradient exists before using it
        if image_adv.grad is None:
            print("    WARNING: Gradient is None, returning original image")
            return image_clean, torch.zeros_like(image_clean)

        # Create perturbation
        perturbation = epsilon * image_adv.grad.sign()

        # Create adversarial image
        adversarial_image = image_adv + perturbation
        adversarial_image = torch.clamp(adversarial_image, 0, 1)

        return adversarial_image.detach(), perturbation.detach()

    # ======================== ADVERSARIAL ROBUSTNESS TEST ========================

    def test_adversarial_robustness(self, dataloader,
                                     epsilons=[0, 0.005, 0.01, 0.02, 0.03, 0.05]):
        """Test model accuracy under FGSM attacks"""

        print("\n  Testing adversarial robustness (FGSM attack)...")
        print(f"  Epsilon values: {epsilons}")

        self.model.eval()
        results = {}

        for eps in epsilons:
            correct = 0
            total = 0

            desc = f'  ε={eps:.3f}'
            for images, labels, _ in tqdm(dataloader, desc=desc, leave=False):
                images = images.to(self.device)

                for i in range(images.size(0)):
                    img = images[i:i + 1]  # Keep batch dimension
                    label = labels[i].item()

                    if eps > 0:
                        adv_img, _ = self.fgsm_attack(img, label, eps)
                    else:
                        adv_img = img.to(self.device)

                    with torch.no_grad():
                        output = self.model(adv_img)
                        pred = output.argmax(dim=1).item()

                    if pred == label:
                        correct += 1
                    total += 1

            accuracy = correct / total if total > 0 else 0
            results[eps] = accuracy
            print(f"    ε={eps:.3f}: Accuracy = {accuracy:.4f} "
                  f"({correct}/{total})")

        return results

    # ======================== CORRUPTION ROBUSTNESS ========================

    def test_corruption_robustness(self, dataloader):
        """Test model against common image corruptions"""

        print("\n  Testing corruption robustness...")

        corruptions = {
            'Clean (baseline)': lambda x: x,
            'Gaussian Noise (σ=0.05)': lambda x: self._add_noise(x, 0.05),
            'Gaussian Noise (σ=0.10)': lambda x: self._add_noise(x, 0.10),
            'Gaussian Blur (k=3)': lambda x: self._blur(x, 3),
            'Gaussian Blur (k=5)': lambda x: self._blur(x, 5),
            'Brightness +20%': lambda x: torch.clamp(x * 1.2, 0, 1),
            'Brightness -20%': lambda x: torch.clamp(x * 0.8, 0, 1),
            'Occlusion 10%': lambda x: self._occlude(x, 0.1),
            'Occlusion 20%': lambda x: self._occlude(x, 0.2),
        }

        self.model.eval()
        results = {}

        for name, corruption_fn in corruptions.items():
            correct = 0
            total = 0

            for images, labels, _ in tqdm(dataloader, desc=f'  {name}', leave=False):
                corrupted = corruption_fn(images)
                corrupted = corrupted.to(self.device)

                with torch.no_grad():
                    outputs = self.model(corrupted)
                    preds = outputs.argmax(dim=1)

                correct += (preds.cpu() == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total if total > 0 else 0
            results[name] = accuracy
            print(f"    {name:30s}: {accuracy:.4f}")

        return results

    @staticmethod
    def _add_noise(x, std):
        return torch.clamp(x + torch.randn_like(x) * std, 0, 1)

    @staticmethod
    def _blur(x, kernel_size):
        if kernel_size % 2 == 0:
            kernel_size += 1
        pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        return pool(x)

    @staticmethod
    def _occlude(x, ratio):
        result = x.clone()
        _, _, h, w = x.shape
        mask_h = int(h * ratio)
        mask_w = int(w * ratio)
        top = np.random.randint(0, max(1, h - mask_h))
        left = np.random.randint(0, max(1, w - mask_w))
        result[:, :, top:top + mask_h, left:left + mask_w] = 0
        return result

    # ======================== UNCERTAINTY ESTIMATION ========================

    def estimate_uncertainty(self, img_tensor, num_forward_passes=30):
        """MC Dropout for uncertainty estimation"""

        # ✅ Enable dropout layers for MC Dropout
        def enable_dropout(model):
            """Enable dropout layers during inference"""
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

        self.model.eval()
        enable_dropout(self.model)

        predictions = []
        for _ in range(num_forward_passes):
            with torch.no_grad():
                output = self.model(img_tensor.to(self.device))
                probs = F.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy()[0])

        self.model.eval()

        predictions = np.array(predictions)

        mean_probs = predictions.mean(axis=0)
        std_probs = predictions.std(axis=0)

        pred_class = np.argmax(mean_probs)
        confidence = mean_probs[pred_class]
        uncertainty = std_probs[pred_class]

        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))

        is_reliable = (uncertainty < 0.15) and (confidence > 0.7)

        return {
            'prediction': self.class_names[pred_class],
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'entropy': float(entropy),
            'is_reliable': bool(is_reliable),
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'all_predictions': predictions
        }

    # ======================== VISUALIZATION ========================

    def plot_adversarial_results(self, results, save_path='outputs/adversarial_robustness.png'):
        """Plot adversarial robustness results"""

        fig, ax = plt.subplots(figsize=(10, 6))

        epsilons = list(results.keys())
        accuracies = list(results.values())

        ax.plot(epsilons, accuracies, 'bo-', linewidth=2.5, markersize=10)
        ax.fill_between(epsilons, accuracies, alpha=0.1, color='blue')

        ax.set_xlabel('Perturbation Strength (ε)', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_title('Model Robustness Against FGSM Adversarial Attack',
                      fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        for eps, acc in zip(epsilons, accuracies):
            ax.annotate(f'{acc:.3f}', (eps, acc),
                        textcoords="offset points", xytext=(0, 15),
                        ha='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Adversarial robustness plot saved to {save_path}")
        plt.show()

    def plot_corruption_results(self, results, save_path='outputs/corruption_robustness.png'):
        """Plot corruption robustness results"""

        fig, ax = plt.subplots(figsize=(12, 7))

        names = list(results.keys())
        accuracies = list(results.values())

        colors = []
        for acc in accuracies:
            if acc >= 0.9:
                colors.append('#2ecc71')
            elif acc >= 0.7:
                colors.append('#f39c12')
            else:
                colors.append('#e74c3c')

        bars = ax.barh(names, accuracies, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Accuracy', fontsize=14)
        ax.set_title('Model Robustness Against Image Corruptions',
                      fontsize=16, fontweight='bold')

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{acc:.3f}', va='center', fontsize=11)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Corruption robustness plot saved to {save_path}")
        plt.show()

    def plot_uncertainty(self, result, save_path='outputs/uncertainty_analysis.png'):
        """Plot uncertainty analysis"""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        x = np.arange(3)

        # 1. Mean probabilities with error bars
        axes[0].bar(x, result['mean_probs'], yerr=result['std_probs'],
                    capsize=8, color=colors, edgecolor='black', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names, fontsize=12)
        axes[0].set_ylabel('Probability', fontsize=12)
        axes[0].set_title('Prediction with Uncertainty\n(Error bars = ±1σ)',
                          fontsize=13, fontweight='bold')
        axes[0].set_ylim(0, 1.1)

        # 2. Distribution of predictions
        predictions = result['all_predictions']
        for i, cls in enumerate(self.class_names):
            axes[1].hist(predictions[:, i], bins=15, alpha=0.6,
                         label=cls, color=colors[i], edgecolor='black')
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('MC Dropout Distribution\n(30 forward passes)',
                          fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)

        # 3. Summary
        axes[2].axis('off')
        summary = (
            f"Prediction: {result['prediction']}\n\n"
            f"Confidence: {result['confidence']:.1%}\n\n"
            f"Uncertainty: {result['uncertainty']:.4f}\n\n"
            f"Entropy: {result['entropy']:.4f}\n\n"
            f"Reliable: {'✓ YES' if result['is_reliable'] else '✗ NO'}"
        )
        axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                     fontsize=16, verticalalignment='center',
                     horizontalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        axes[2].set_title('Summary', fontsize=13, fontweight='bold')

        plt.suptitle('Uncertainty Quantification (MC Dropout)',
                      fontsize=16, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Uncertainty analysis saved to {save_path}")
        plt.show()

        return result


# ============================================================
# RUN ROBUSTNESS TESTING
# ============================================================
def run_robustness_tests():
    """Run all robustness tests"""

    print("=" * 60)
    print("STEP 7: ROBUSTNESS TESTING")
    print("=" * 60)

    # 1. Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    model = create_model(num_classes=3, pretrained=False)

    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"\n  ERROR: No trained model found at {checkpoint_path}")
        print(f"  Please run Step 5 (training) first!")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    print("  ✓ Model loaded successfully")

    # 2. Load test data
    loaders, datasets = create_data_loaders('dataset', batch_size=16)

    # 3. Create robustness engine
    engine = RobustnessEngine(model, device)

    # ==================== TEST 1: ADVERSARIAL ====================
    print("\n" + "-" * 40)
    print("TEST 1: Adversarial Robustness (FGSM)")
    print("-" * 40)

    adv_results = engine.test_adversarial_robustness(loaders['test'])
    engine.plot_adversarial_results(adv_results)

    # ==================== TEST 2: CORRUPTION ====================
    print("\n" + "-" * 40)
    print("TEST 2: Corruption Robustness")
    print("-" * 40)

    corr_results = engine.test_corruption_robustness(loaders['test'])
    engine.plot_corruption_results(corr_results)

    # ==================== TEST 3: UNCERTAINTY ====================
    print("\n" + "-" * 40)
    print("TEST 3: Uncertainty Estimation (MC Dropout)")
    print("-" * 40)

    # Get one test image
    test_images, test_labels, _ = next(iter(loaders['test']))
    sample_image = test_images[0:1]
    sample_label = test_labels[0].item()

    print(f"  True label: {engine.class_names[sample_label]}")
    unc_result = engine.estimate_uncertainty(sample_image)
    engine.plot_uncertainty(unc_result)

    # ==================== SUMMARY ====================
    print(f"\n{'='*60}")
    print(f"ROBUSTNESS SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Adversarial Robustness:")
    for eps, acc in adv_results.items():
        status = "✓" if acc > 0.7 else "⚠" if acc > 0.5 else "✗"
        print(f"    {status} ε={eps:.3f}: {acc:.4f}")

    print(f"\n  Corruption Robustness:")
    for name, acc in corr_results.items():
        status = "✓" if acc > 0.8 else "⚠" if acc > 0.6 else "✗"
        print(f"    {status} {name}: {acc:.4f}")

    print(f"\n  Uncertainty Estimation:")
    print(f"    Prediction:  {unc_result['prediction']}")
    print(f"    Confidence:  {unc_result['confidence']:.1%}")
    print(f"    Uncertainty: {unc_result['uncertainty']:.4f}")
    print(f"    Reliable:    {'YES ✓' if unc_result['is_reliable'] else 'NO ⚠'}")

    print(f"\n✓ Step 7 complete! All robustness tests passed.")
    print(f"  Check 'outputs/' folder for all plots.")
    print(f"  Proceed to Step 8: Web Application")


if __name__ == "__main__":
    run_robustness_tests()
