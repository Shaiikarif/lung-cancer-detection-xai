"""
Clean Dataset Splitting Script for PulmoAI.
Generates leakage-free, stratified train/val/test splits (70/15/15) from the raw
IQ-OTH/NCCD dataset.
Ensures identical SHA-256 hashes are grouped into the same split (zero cross-split overlap).
Exports train_manifest.csv, validation_manifest.csv, test_manifest.csv, split_summary.json,
split_integrity_report.md, and creates clean data folders.
"""

import os
import sys
import shutil
import hashlib
import json
import csv
import random
from pathlib import Path
from collections import defaultdict

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def create_clean_splits(root_dir: Path, output_base: Path, seed: int = 42):
    random.seed(seed)

    raw_dir = root_dir / "The IQ-OTHNCCD lung cancer dataset"
    splits_dir = output_base / "clean_splits"
    data_dir = splits_dir / "data"

    splits_dir.mkdir(parents=True, exist_ok=True)

    # Clean output data directory if exists to ensure idempotency
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        for cls in ["Benign", "Malignant", "Normal"]:
            (data_dir / sp / cls).mkdir(parents=True, exist_ok=True)

    raw_classes = {
        "Benign cases": "Benign",
        "Malignant cases": "Malignant",
        "Normal cases": "Normal"
    }

    # Group files by class and by hash
    # class_name -> hash -> list of file records
    class_hash_map = defaultdict(lambda: defaultdict(list))
    all_raw_files = []

    for folder_name, cls_name in raw_classes.items():
        folder_path = raw_dir / folder_name
        for f in sorted(folder_path.iterdir()):
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                sha = compute_sha256(f)
                rec = {
                    "source_path": str(f.resolve()),
                    "relative_raw_path": str(f.relative_to(root_dir)).replace("\\", "/"),
                    "filename": f.name,
                    "class_name": cls_name,
                    "sha256": sha,
                    "size_bytes": f.stat().st_size
                }
                class_hash_map[cls_name][sha].append(rec)
                all_raw_files.append(rec)

    split_manifests = {
        "train": [],
        "val": [],
        "test": []
    }

    # For each class, stratify unique hash clusters (70% train, 15% val, 15% test)
    for cls_name, hash_dict in class_hash_map.items():
        unique_hashes = list(hash_dict.keys())
        # Shuffle hashes deterministically
        random.shuffle(unique_hashes)

        # Count total items across hashes to match target image counts
        total_items = sum(len(hash_dict[h]) for h in unique_hashes)
        target_train = int(round(0.70 * total_items))
        target_val = int(round(0.15 * total_items))
        # remaining is test

        current_train, current_val = 0, 0

        train_hashes, val_hashes, test_hashes = set(), set(), set()

        for h in unique_hashes:
            items_count = len(hash_dict[h])
            if current_train + items_count <= target_train:
                train_hashes.add(h)
                current_train += items_count
            elif current_val + items_count <= target_val:
                val_hashes.add(h)
                current_val += items_count
            else:
                test_hashes.add(h)

        # Assign records
        for h in train_hashes:
            for r in hash_dict[h]:
                rec = dict(r)
                rec["split"] = "train"
                rec["split_path"] = f"research_audit/clean_splits/data/train/{cls_name}/{rec['filename']}"
                split_manifests["train"].append(rec)
                dest = data_dir / "train" / cls_name / rec["filename"]
                shutil.copy2(rec["source_path"], dest)

        for h in val_hashes:
            for r in hash_dict[h]:
                rec = dict(r)
                rec["split"] = "val"
                rec["split_path"] = f"research_audit/clean_splits/data/val/{cls_name}/{rec['filename']}"
                split_manifests["val"].append(rec)
                dest = data_dir / "val" / cls_name / rec["filename"]
                shutil.copy2(rec["source_path"], dest)

        for h in test_hashes:
            for r in hash_dict[h]:
                rec = dict(r)
                rec["split"] = "test"
                rec["split_path"] = f"research_audit/clean_splits/data/test/{cls_name}/{rec['filename']}"
                split_manifests["test"].append(rec)
                dest = data_dir / "test" / cls_name / rec["filename"]
                shutil.copy2(rec["source_path"], dest)

    # Verification: check cross-split hash overlap
    train_h = set(r["sha256"] for r in split_manifests["train"])
    val_h = set(r["sha256"] for r in split_manifests["val"])
    test_h = set(r["sha256"] for r in split_manifests["test"])

    assert len(train_h.intersection(val_h)) == 0, "Leakage detected: train and val share hashes!"
    assert len(train_h.intersection(test_h)) == 0, "Leakage detected: train and test share hashes!"
    assert len(val_h.intersection(test_h)) == 0, "Leakage detected: val and test share hashes!"

    total_split_images = len(split_manifests["train"]) + len(split_manifests["val"]) + len(split_manifests["test"])
    assert total_split_images == len(all_raw_files), f"Image count mismatch: {total_split_images} != {len(all_raw_files)}"

    print("Splits generated successfully with ZERO hash overlap across all splits!")
    print(f"Total images: {total_split_images}")
    for sp in ["train", "val", "test"]:
        print(f"  {sp}: {len(split_manifests[sp])} images")

    # Write CSV manifests
    manifest_fields = ["filename", "class_name", "split", "sha256", "size_bytes", "split_path", "relative_raw_path", "source_path"]

    for sp, fname in [("train", "train_manifest.csv"), ("val", "validation_manifest.csv"), ("test", "test_manifest.csv")]:
        p = splits_dir / fname
        with open(p, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=manifest_fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(split_manifests[sp])
        print(f"Manifest written: {p}")

    # Generate split_summary.json
    summary = {
        "random_seed": seed,
        "total_images": total_split_images,
        "unique_hashes": len(set(r["sha256"] for r in all_raw_files)),
        "split_counts": {
            sp: len(split_manifests[sp]) for sp in ["train", "val", "test"]
        },
        "per_class_counts": {
            sp: {
                cls: sum(1 for r in split_manifests[sp] if r["class_name"] == cls)
                for cls in ["Benign", "Malignant", "Normal"]
            }
            for sp in ["train", "val", "test"]
        },
        "hash_leakage_check": {
            "train_val_overlap": len(train_h.intersection(val_h)),
            "train_test_overlap": len(train_h.intersection(test_h)),
            "val_test_overlap": len(val_h.intersection(test_h)),
            "leakage_free": True
        }
    }

    with open(splits_dir / "split_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"Split summary written: {splits_dir / 'split_summary.json'}")

    # Generate split_integrity_report.md
    with open(splits_dir / "split_integrity_report.md", 'w', encoding='utf-8') as f:
        f.write("# Clean Dataset Split & Partition Integrity Report\n\n")
        f.write("## 1. Split Specification\n\n")
        f.write(f"- **Source Dataset:** Raw `The IQ-OTHNCCD lung cancer dataset`\n")
        f.write(f"- **Partition Ratios:** 70% Train / 15% Validation / 15% Test (Stratified)\n")
        f.write(f"- **Deduplication Strategy:** Hash clustering. Identical SHA-256 hashes are placed in the same partition.\n")
        f.write(f"- **Random Seed:** `{seed}`\n")
        f.write(f"- **Total Images:** `{total_split_images}`\n")
        f.write(f"- **Total Unique Hashes:** `{len(train_h) + len(val_h) + len(test_h)}`\n\n")

        f.write("## 2. Partition Summary Table\n\n")
        f.write("| Partition | Benign | Malignant | Normal | Total Images | Total Hashes | Percentage |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for sp in ["train", "val", "test"]:
            b = summary["per_class_counts"][sp]["Benign"]
            m = summary["per_class_counts"][sp]["Malignant"]
            n = summary["per_class_counts"][sp]["Normal"]
            tot = summary["split_counts"][sp]
            h_count = len(set(r["sha256"] for r in split_manifests[sp]))
            pct = round(tot / total_split_images * 100, 2)
            f.write(f"| **{sp.capitalize()}** | {b} | {m} | {n} | {tot} | {h_count} | {pct}% |\n")
        f.write(f"| **Total** | **120** | **561** | **416** | **{total_split_images}** | **{len(train_h)+len(val_h)+len(test_h)}** | **100.00%** |\n\n")

        f.write("## 3. Cross-Partition Overlap Verification\n\n")
        f.write("| Overlap Verification Pair | Shared SHA-256 Hashes | Shared Files | Status |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Train vs. Validation | `{len(train_h.intersection(val_h))}` | `0` | PASS (Zero Leakage) |\n")
        f.write(f"| Train vs. Test | `{len(train_h.intersection(test_h))}` | `0` | PASS (Zero Leakage) |\n")
        f.write(f"| Validation vs. Test | `{len(val_h.intersection(test_h))}` | `0` | PASS (Zero Leakage) |\n\n")

        f.write("## 4. Manifest Artifacts\n\n")
        f.write("- `research_audit/clean_splits/train_manifest.csv`\n")
        f.write("- `research_audit/clean_splits/validation_manifest.csv`\n")
        f.write("- `research_audit/clean_splits/test_manifest.csv`\n")
        f.write("- `research_audit/clean_splits/split_summary.json`\n")
        f.write("- Clean data folder: `research_audit/clean_splits/data/{train,val,test}/{Benign,Malignant,Normal}`\n")

    print(f"Split integrity report written: {splits_dir / 'split_integrity_report.md'}")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    create_clean_splits(root, root / "research_audit")
