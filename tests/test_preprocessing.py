"""Unit tests for preprocessing transforms and tensor normalization."""

import unittest
import numpy as np
import torch

from src.preprocessing.transforms import (
    get_train_transforms,
    get_eval_transforms,
    normalize_tensor,
    unnormalize_tensor,
    DEFAULT_MEAN,
    DEFAULT_STD
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.img_size = 224
        self.dummy_rgb = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

    def test_eval_transform_shape(self):
        transform = get_eval_transforms(img_size=self.img_size)
        res = transform(image=self.dummy_rgb)
        tensor = res["image"]
        self.assertEqual(tensor.shape, (3, self.img_size, self.img_size))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_train_transform_shape(self):
        transform = get_train_transforms(img_size=self.img_size)
        res = transform(image=self.dummy_rgb)
        tensor = res["image"]
        self.assertEqual(tensor.shape, (3, self.img_size, self.img_size))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_unnormalize_tensor_bounds(self):
        # Create a pixel tensor in [0, 1]
        x_raw = torch.rand(2, 3, 64, 64)
        # Normalize
        x_norm = normalize_tensor(x_raw, DEFAULT_MEAN, DEFAULT_STD)
        # Unnormalize
        x_reconstructed = unnormalize_tensor(x_norm, DEFAULT_MEAN, DEFAULT_STD)

        self.assertTrue(torch.all(x_reconstructed >= 0.0))
        self.assertTrue(torch.all(x_reconstructed <= 1.0))
        # Check round-trip accuracy
        self.assertTrue(torch.allclose(x_raw, x_reconstructed, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
