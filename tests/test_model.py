"""Unit tests for model architecture and attention mechanisms."""

import unittest
from pathlib import Path
import torch

from src.models.attention import AttentionModule
from src.models.resnet50 import LungCancerModel, create_model


class TestModelArchitecture(unittest.TestCase):

    def test_attention_module_shapes(self):
        batch, channels, h, w = 2, 2048, 7, 7
        att = AttentionModule(in_channels=channels, reduction=16)
        x = torch.randn(batch, channels, h, w)
        out = att(x)

        self.assertEqual(out.shape, (batch, channels, h, w))
        self.assertIsNotNone(att.last_spatial_weights)
        self.assertEqual(att.last_spatial_weights.shape, (batch, 1, h, w))

    def test_lung_cancer_model_forward(self):
        model = create_model(num_classes=3, pretrained=False, dropout_rate=0.5)
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (2, 3))
        self.assertFalse(torch.isnan(output).any())

    def test_checkpoint_compatibility(self):
        ckpt_path = Path("checkpoints/best_model.pth")
        if not ckpt_path.exists():
            self.skipTest("best_model.pth checkpoint not present on disk.")

        model = create_model(num_classes=3, pretrained=False)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Verify strict key matching without missing or unexpected keys
        load_result = model.load_state_dict(state, strict=True)
        self.assertEqual(len(load_result.missing_keys), 0)
        self.assertEqual(len(load_result.unexpected_keys), 0)


if __name__ == "__main__":
    unittest.main()
