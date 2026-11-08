"""Unit tests for Explainability components (Grad-CAM and Integrated Gradients)."""

import unittest
import numpy as np
import torch

from src.models.resnet50 import create_model
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.integrated_gradients import IntegratedGradientsExplainer
from src.preprocessing.transforms import normalize_tensor, DEFAULT_MEAN, DEFAULT_STD


class TestExplainability(unittest.TestCase):

    def setUp(self):
        self.model = create_model(num_classes=3, pretrained=False)
        self.model.eval()

    def test_gradcam_explainer(self):
        explainer = GradCAMExplainer(self.model)
        raw_float = np.random.rand(224, 224, 3).astype(np.float32)
        raw_tensor = torch.from_numpy(raw_float.transpose(2, 0, 1)).unsqueeze(0)
        norm_tensor = normalize_tensor(raw_tensor, DEFAULT_MEAN, DEFAULT_STD)

        res = explainer.explain(norm_tensor, raw_float, target_class=0, use_plus_plus=False)

        self.assertEqual(res["heatmap"].shape, (224, 224))
        self.assertEqual(res["overlay"].shape, (224, 224, 3))
        self.assertGreaterEqual(res["heatmap"].min(), 0.0)
        self.assertLessEqual(res["heatmap"].max(), 1.0)

    def test_integrated_gradients_explainer(self):
        explainer = IntegratedGradientsExplainer(self.model)
        raw_float = np.random.rand(224, 224, 3).astype(np.float32)
        raw_tensor = torch.from_numpy(raw_float.transpose(2, 0, 1)).unsqueeze(0)
        norm_tensor = normalize_tensor(raw_tensor, DEFAULT_MEAN, DEFAULT_STD)

        res = explainer.explain(norm_tensor, target_class=0, steps=5)

        self.assertEqual(res["attribution_map"].shape, (224, 224))
        self.assertGreaterEqual(res["attribution_map"].min(), 0.0)
        self.assertLessEqual(res["attribution_map"].max(), 1.0)


if __name__ == "__main__":
    unittest.main()
