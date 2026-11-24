"""Unit tests for clean dataset splitting and leak-free partition verification."""

import unittest
import csv
from pathlib import Path


class TestCleanSplitsIntegrity(unittest.TestCase):
    """Verifies that clean splits generated in research_audit/clean_splits are leakage-free."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.clean_dir = cls.root / "research_audit" / "clean_splits"
        cls.train_manifest = cls.clean_dir / "train_manifest.csv"
        cls.val_manifest = cls.clean_dir / "validation_manifest.csv"
        cls.test_manifest = cls.clean_dir / "test_manifest.csv"

    def test_manifests_exist(self):
        self.assertTrue(self.train_manifest.exists(), "train_manifest.csv does not exist")
        self.assertTrue(self.val_manifest.exists(), "validation_manifest.csv does not exist")
        self.assertTrue(self.test_manifest.exists(), "test_manifest.csv does not exist")

    def test_zero_hash_overlap(self):
        def get_hashes(manifest_path):
            hashes = set()
            with open(manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    hashes.add(row["sha256"])
            return hashes

        train_hashes = get_hashes(self.train_manifest)
        val_hashes = get_hashes(self.val_manifest)
        test_hashes = get_hashes(self.test_manifest)

        self.assertEqual(len(train_hashes & val_hashes), 0, "Train and Val share hashes!")
        self.assertEqual(len(train_hashes & test_hashes), 0, "Train and Test share hashes!")
        self.assertEqual(len(val_hashes & test_hashes), 0, "Val and Test share hashes!")

    def test_total_image_count_1097(self):
        total = 0
        for p in [self.train_manifest, self.val_manifest, self.test_manifest]:
            with open(p, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                total += sum(1 for _ in reader)
        self.assertEqual(total, 1097, f"Expected 1,097 images, got {total}")


if __name__ == "__main__":
    unittest.main()
