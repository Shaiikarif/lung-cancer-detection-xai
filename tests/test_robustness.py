"""Unit tests for robustness modules: input validation, FGSM, and MC Dropout."""

import unittest
import numpy as np
import torch

from src.models.resnet50 import create_model
from src.robustness.input_validation import InputValidator
from src.robustness.adversarial import AdversarialEvaluator
from src.robustness.uncertainty import UncertaintyQuantifier
from src.preprocessing.transforms import normalize_tensor, DEFAULT_MEAN, DEFAULT_STD


class TestRobustness(unittest.TestCase):

    def setUp(self):
        self.model = create_model(num_classes=3, pretrained=False)
        self.model.eval()

    def test_input_validator_degenerate_image(self):
        # Blank / uniform image should fail validation
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        res = InputValidator.validate_numpy_image(blank)
        self.assertFalse(res.is_valid)
        self.assertIn("degenerate", res.error_message.lower())

    def test_input_validator_small_image(self):
        # Image smaller than 64x64 should fail
        tiny = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        res = InputValidator.validate_numpy_image(tiny)
        self.assertFalse(res.is_valid)

    def test_input_validator_valid_image(self):
        valid = np.random.randint(20, 220, (256, 256, 3), dtype=np.uint8)
        res = InputValidator.validate_numpy_image(valid)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.width, 256)
        self.assertEqual(res.height, 256)

    def test_fgsm_attack_bounds(self):
        evaluator = AdversarialEvaluator(self.model)
        # Normalized input tensor
        clean_raw = torch.rand(1, 3, 224, 224)
        clean_norm = normalize_tensor(clean_raw, DEFAULT_MEAN, DEFAULT_STD)

        adv_norm, perturbation = evaluator.fgsm_attack(clean_norm, target_label=0, epsilon=0.03)

        self.assertEqual(adv_norm.shape, clean_norm.shape)
        self.assertEqual(perturbation.shape, clean_raw.shape)
        # Check that perturbation is within +/- epsilon
        self.assertTrue(torch.all(perturbation.abs() <= 0.030001))

    def test_uncertainty_quantifier(self):
        uq = UncertaintyQuantifier(self.model)
        clean_norm = normalize_tensor(torch.rand(1, 3, 224, 224), DEFAULT_MEAN, DEFAULT_STD)
        res = uq.estimate(clean_norm, num_samples=5)

        self.assertIn("predicted_class", res)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertGreaterEqual(res["uncertainty"], 0.0)
        self.assertGreaterEqual(res["entropy"], 0.0)
        self.assertIsInstance(res["is_reliable"], bool)


if __name__ == "__main__":
    unittest.main()
