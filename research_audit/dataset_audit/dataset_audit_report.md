# Forensic Dataset Audit & Data Leakage Investigation Report

## 1. Executive Summary

A deep forensic audit of the local IQ-OTH/NCCD Lung Cancer CT dataset was performed. The audit evaluated exact SHA-256 image hashes, pixel dimensions, file corruption, partition overlaps, and slice-vs-patient metadata across both the raw archive and the historical `dataset/` partitions.

### Critical Finding: Severe Historical Data Leakage
- **Raw Unique Images:** 1097 images across 3 classes (120 Benign, 561 Malignant, 416 Normal).
- **Historical Partition Total:** 1549 files (968 train, 292 val, 289 test) - an accumulation of 452 surplus copies.
- **Train-Test Hash Collision:** **199 unique image hashes** in the test set were also present in the training set.
- **Per-Class Leakage in Historical Test Set:**
  - **Normal**: 98 / 122 test images (80.33%) were exact duplicates of training images!
  - **Malignant**: 98 / 149 test images (65.77%) were exact duplicates of training images!
  - **Benign**: 4 / 18 test images (22.22%) were exact duplicates of training images!

## 2. Root Cause Analysis

### A. Non-Idempotent Split Accumulation
In historical iterations of `step2_prepare_data.py` / `prepare_data.py`, `shutil.copy2()` was invoked without first wiping or synchronizing the destination directories (`dataset/train`, `dataset/val`, `dataset/test`). Consequently, whenever data preparation was re-run with different seeds or shuffle orders, images previously assigned to `train` were copied into `test` while their earlier copies remained in `train`.

### B. Direct Consequence on Reported Metrics
This explains the historically reported metrics in `outputs/final_results.json`:
- Malignant class achieved **1.0000 (100%) Precision, Recall, and F1** because 65.8% of test images were already seen during training.
- Normal class achieved **0.9918 (99.2%) Recall** because 80.3% of test images were already seen during training.
- Benign class had 0% leakage (0 duplicate copies between train and test), which explains why Benign recall was noticeably lower (88.9%) than Normal and Malignant in historical runs.

## 3. Image Integrity & Dimensions

- **Corrupted Images:** 0 (Zero corrupted files detected).
- **Raw Image Dimensions:**
  - `512x512 (3ch)`: 1036 images
  - `511x404 (3ch)`: 1 images
  - `801x512 (3ch)`: 28 images
  - `623x512 (3ch)`: 31 images
  - `506x331 (3ch)`: 1 images

## 4. Patient Independence & De-Identification Forensic

The dataset documentation (`IQ-OTH_NCCD lung cancer dataset.txt`) notes that the collection comprises 110 patients (40 malignant, 15 benign, 55 normal). However, in the Kaggle release distribution:
1. All DICOM metadata was stripped, and images were converted to standard 8-bit JPEG files.
2. Filenames follow flat class-level indices: `Bengin case (1).jpg` to `Bengin case (120).jpg`, `Malignant case (1).jpg` to `Malignant case (561).jpg`, and `Normal case (1).jpg` to `Normal case (416).jpg`.
3. There are no patient identifier tags, case boundary markers, or slice position headers in the file bytes or names.
4. As a result, **patient-level grouping is not mathematically recoverable from the Kaggle JPEG files alone** without external clinical mapping tables.
5. Slices must therefore be treated honestly as de-identified 2D axial CT slices. To prevent data leakage, stratified splitting must strictly ensure that **no slice or duplicate image hash appears in multiple splits**.

## 5. Corrective Action Plan

1. Clean the split pipeline to be strictly idempotent (`shutil.rmtree` or directory wipe before writing).
2. Generate fresh, non-overlapping 70/15/15 stratified splits sourced purely from `The IQ-OTHNCCD lung cancer dataset`.
3. Save manifests (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) with SHA-256 verification.
4. Retrain the ResNet-50 + CBAM architecture on the clean split.
5. Evaluate on the clean, unseen test split to provide authentic, scientifically rigorous benchmark metrics.
