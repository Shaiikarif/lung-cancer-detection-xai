# step6_explainability.py
"""
STEP 6: Explainability Module (FIXED VERSION)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lime import lime_image
from skimage.segmentation import mark_boundaries

from step4_model import create_model


class ExplainabilityEngine:
    """Engine that generates visual explanations for model predictions."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

        self.class_names = ['Normal', 'Benign', 'Malignant']
        self.class_colors = ['#2ecc71', '#f39c12', '#e74c3c']

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def load_image(self, image_path, img_size=224):
        """Load and preprocess an image"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))

        img_display = img.astype(np.float32) / 255.0

        img_normalized = (img_display - self.mean) / self.std
        img_tensor = torch.from_numpy(
            img_normalized.transpose(2, 0, 1)
        ).float().unsqueeze(0).to(self.device)

        return img, img_display, img_tensor

    def predict(self, img_tensor):
        """Get prediction from model"""
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        return predicted_class, confidence, probabilities.cpu().numpy()

    # ======================== GRAD-CAM ========================

    def generate_gradcam(self, image_path):
        """Generate Grad-CAM heatmap"""

        print("\n  Generating Grad-CAM...")

        img, img_display, img_tensor = self.load_image(image_path)
        pred_class, confidence, probs = self.predict(img_tensor)

        # Target layer: last convolutional layer of ResNet50
        target_layer = self.model.backbone.layer4[-1]

        # Create Grad-CAM
        cam = GradCAM(model=self.model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(pred_class)]

        # Generate heatmap
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]

        # Overlay heatmap on original image
        overlay = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

        # ✅ FIX: Use del instead of cam.release()
        del cam

        print(f"    Prediction: {self.class_names[pred_class]} ({confidence:.1%})")

        return {
            'original': img,
            'heatmap': grayscale_cam,
            'overlay': overlay,
            'prediction': self.class_names[pred_class],
            'confidence': confidence,
            'probabilities': probs
        }

    def generate_gradcam_plusplus(self, image_path):
        """Generate Grad-CAM++ (improved version)"""

        print("  Generating Grad-CAM++...")

        img, img_display, img_tensor = self.load_image(image_path)
        pred_class, confidence, probs = self.predict(img_tensor)

        target_layer = self.model.backbone.layer4[-1]
        cam = GradCAMPlusPlus(model=self.model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(pred_class)]

        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
        overlay = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

        # ✅ FIX: Use del instead of cam.release()
        del cam

        return {
            'original': img,
            'heatmap': grayscale_cam,
            'overlay': overlay,
            'prediction': self.class_names[pred_class],
            'confidence': confidence,
            'probabilities': probs
        }

    # ======================== LIME ========================

    def generate_lime(self, image_path, num_samples=500):
        """Generate LIME explanation"""

        print("  Generating LIME explanation...")

        img, img_display, img_tensor = self.load_image(image_path)
        pred_class, confidence, probs = self.predict(img_tensor)

        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            batch = []
            for image in images:
                normalized = (image.astype(np.float32) / 255.0 - self.mean) / self.std
                tensor = torch.from_numpy(
                    normalized.transpose(2, 0, 1)
                ).float()
                batch.append(tensor)

            batch_tensor = torch.stack(batch).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)

            return probabilities.cpu().numpy()

        explanation = explainer.explain_instance(
            img,
            predict_fn,
            top_labels=3,
            hide_color=0,
            num_samples=num_samples
        )

        # Positive features
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        lime_positive = mark_boundaries(temp / 255.0, mask, color=(0, 1, 0))

        # All features
        temp2, mask2 = explanation.get_image_and_mask(
            pred_class,
            positive_only=False,
            num_features=10,
            hide_rest=False
        )
        lime_full = mark_boundaries(temp2 / 255.0, mask2, color=(1, 0, 0))

        print(f"    Prediction: {self.class_names[pred_class]} ({confidence:.1%})")

        return {
            'original': img,
            'lime_positive': lime_positive,
            'lime_full': lime_full,
            'prediction': self.class_names[pred_class],
            'confidence': confidence,
            'probabilities': probs
        }

    # ======================== INTEGRATED GRADIENTS ========================

    def generate_integrated_gradients(self, image_path, steps=50):
        """Generate Integrated Gradients explanation"""

        print("  Generating Integrated Gradients...")

        img, img_display, img_tensor = self.load_image(image_path)
        pred_class, confidence, probs = self.predict(img_tensor)

        baseline = torch.zeros_like(img_tensor).to(self.device)

        alphas = torch.linspace(0, 1, steps + 1).to(self.device)

        all_gradients = []

        for alpha in alphas:
            interpolated = baseline + alpha * (img_tensor - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            output = self.model(interpolated)
            target_score = output[0, pred_class]

            self.model.zero_grad()
            target_score.backward()

            all_gradients.append(interpolated.grad.cpu().detach())

        avg_gradients = torch.mean(torch.stack(all_gradients), dim=0)

        integrated_grads = (img_tensor.cpu() - baseline.cpu()) * avg_gradients
        integrated_grads = integrated_grads.squeeze().numpy()

        attribution = np.mean(np.abs(integrated_grads), axis=0)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)

        return {
            'original': img,
            'attribution_map': attribution,
            'prediction': self.class_names[pred_class],
            'confidence': confidence,
            'probabilities': probs
        }

    # ======================== FULL EXPLANATION ========================

    def explain(self, image_path, save_path=None):
        """Generate COMPLETE explanation with all methods"""

        print(f"\n{'='*60}")
        print(f"Generating Full AI Explanation")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # 1. Grad-CAM
        gradcam = self.generate_gradcam(image_path)

        # 2. Grad-CAM++
        gradcampp = self.generate_gradcam_plusplus(image_path)

        # 3. LIME
        lime_result = self.generate_lime(image_path, num_samples=300)

        # 4. Integrated Gradients
        ig = self.generate_integrated_gradients(image_path, steps=30)

        # ==================== CREATE VISUALIZATION ====================

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))

        pred = gradcam['prediction']
        conf = gradcam['confidence']
        probs = gradcam['probabilities']

        # Row 1
        axes[0, 0].imshow(gradcam['original'])
        axes[0, 0].set_title(
            f'Original CT Scan\nPrediction: {pred} ({conf:.1%})',
            fontsize=13, fontweight='bold'
        )
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gradcam['overlay'])
        axes[0, 1].set_title('Grad-CAM\n(Red = Important)', fontsize=13)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(gradcampp['overlay'])
        axes[0, 2].set_title('Grad-CAM++\n(Enhanced)', fontsize=13)
        axes[0, 2].axis('off')

        im = axes[0, 3].imshow(ig['attribution_map'], cmap='jet')
        axes[0, 3].set_title('Integrated Gradients\n(Pixel Importance)', fontsize=13)
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

        # Row 2
        axes[1, 0].imshow(lime_result['lime_positive'])
        axes[1, 0].set_title('LIME\n(Green = Supporting)', fontsize=13)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(lime_result['lime_full'])
        axes[1, 1].set_title('LIME\n(Full Analysis)', fontsize=13)
        axes[1, 1].axis('off')

        # Probability bar chart
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = axes[1, 2].barh(self.class_names, probs, color=colors, edgecolor='black')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_title('Class Probabilities', fontsize=13, fontweight='bold')
        for bar, p in zip(bars, probs):
            axes[1, 2].text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{p:.3f}', va='center', fontsize=12
            )

        # Heatmap overlay
        axes[1, 3].imshow(gradcam['original'])
        heatmap_colored = cv2.applyColorMap(
            (gradcam['heatmap'] * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        axes[1, 3].imshow(heatmap_colored, alpha=0.5)
        axes[1, 3].set_title('Heatmap Overlay\n(Semi-transparent)', fontsize=13)
        axes[1, 3].axis('off')

        plt.suptitle(
            f'Explainable AI Analysis for Lung Cancer Detection\n'
            f'Diagnosis: {pred} | Confidence: {conf:.1%}',
            fontsize=16, fontweight='bold', y=1.02
        )

        plt.tight_layout()

        if save_path is None:
            save_path = f'outputs/explanation_{pred.lower()}.png'

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Full explanation saved to: {save_path}")
        plt.show()

        return {
            'prediction': pred,
            'confidence': conf,
            'probabilities': {self.class_names[i]: float(probs[i]) for i in range(3)},
        }


# ============================================================
# RUN EXPLAINABILITY
# ============================================================
def run_explainability():
    """Generate explanations for test images"""

    print("=" * 60)
    print("STEP 6: EXPLAINABILITY ANALYSIS")
    print("=" * 60)

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
    model.eval()
    print("  ✓ Model loaded successfully")

    engine = ExplainabilityEngine(model, device)

    test_dir = 'dataset/test'

    for class_name in ['normal', 'benign', 'malignant']:
        class_dir = os.path.join(test_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"\n  ⚠ Skipping {class_name}: directory not found")
            continue

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if images:
            image_path = os.path.join(class_dir, images[0])
            save_path = f'outputs/explanation_{class_name}.png'

            try:
                engine.explain(image_path, save_path=save_path)
            except Exception as e:
                print(f"\n  ✗ Error explaining {class_name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n✓ Step 6 complete! Explanations generated.")
    print(f"  Check the 'outputs/' folder for visualization images.")


if __name__ == "__main__":
    run_explainability()
