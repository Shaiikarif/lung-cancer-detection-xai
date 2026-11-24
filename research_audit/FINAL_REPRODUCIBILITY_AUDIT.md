# PulmoAI Final Scientific & Reproducibility Audit

**Audit Date:** 2026-11-22
**Auditor Role:** Senior ML Engineer & Independent Scientific Reviewer
**Repository:** [lung-cancer-detection-xai](https://github.com/Shaiikarif/lung-cancer-detection-xai)
**Evaluated Checkpoint:** `research_audit/checkpoints/retrained_best_model.pth` (SHA-256 verified)
**Evaluation Dataset:** `research_audit/clean_splits/data/test` ($N=165$ images)

---

## 1. Executive Audit Summary

A comprehensive, read-only scientific integrity and reproducibility audit of the PulmoAI framework was performed. The audit reviewed model provenance, training/validation/test split isolation, raw prediction metrics, test suite health, git repository state, and scientific claims.

### Key Audit Findings
1. **Certified Leakage-Free Splitting:** The historical data leakage—wherein 69.2% of the test partition consisted of exact duplicates of training images—has been fully resolved. Sourced directly from `The IQ-OTHNCCD lung cancer dataset`, the certified train, validation, and test manifests exhibit **strictly 0.0% SHA-256 hash overlap**.
2. **True Out-of-Sample Evaluation:** The test partition was strictly held out. The training script instantiated only `train` and `val` loaders; checkpoint selection was driven solely by validation loss minimization.
3. **Metric Verification:** All reported test metrics (Accuracy: 97.58%, Macro F1: 95.04%, Weighted F1: 97.52%) mathematically match the raw prediction tensors and confusion matrix diagonal down to machine precision.
4. **Regression & Test Suite:** 16 out of 16 automated tests passed cleanly in 16.39 seconds, including model forward passes, CBAM attention tensor shapes, FGSM attack bounds, explainability hooks, and partition integrity assertions.
5. **Preservation of Historical Baselines:** Historical weights (`checkpoints/best_model.pth`, `checkpoints/final_model.pth`) and baseline results (`outputs/final_results.json`) remain untouched.

---

## 2. Partition Isolation & Manifest Audit

The clean dataset was generated using hash clustering with random seed 42 to guarantee that duplicate image hashes in the raw collection never straddle partition boundaries.

### Partition Summary & Hash Isolation

| Partition | Benign | Malignant | Normal | Total Images | Unique Hashes | Manifest Path | Cross-Partition Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Train** | 84 | 393 | 291 | 768 (70.01%) | 743 | `research_audit/clean_splits/train_manifest.csv` | **0 SHA-256 Hashes** |
| **Validation** | 18 | 84 | 62 | 164 (14.95%) | 155 | `research_audit/clean_splits/validation_manifest.csv` | **0 SHA-256 Hashes** |
| **Test** | 18 | 84 | 63 | 165 (15.04%) | 156 | `research_audit/clean_splits/test_manifest.csv` | **0 SHA-256 Hashes** |
| **Total** | **120** | **561** | **416** | **1,097 (100.0%)** | **1,054** | — | **Zero Leakage Verified** |

### Duplicate Group Integrity
* Total duplicate groups detected in raw archive: **43 groups**.
* Audit verification: **100% of duplicate groups are assigned exclusively to a single split**. Zero duplicate hashes exist across train/val, train/test, or val/test.

---

## 3. Checkpoint Provenance & Training Isolation

### Training Execution
* **Script:** `scripts/train_clean.py`
* **Architecture:** ResNet-50 + CBAM Attention (`src/models/resnet50.py`, 25,148,839 parameters)
* **Optimization:** AdamW (`lr=1e-4` to `2.93e-5`, `weight_decay=1e-4`) with `CosineAnnealingWarmRestarts`
* **Loss Function:** Multi-Class Focal Loss (`gamma=2.0`)
* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.29 GB VRAM, CUDA 11.8)
* **Batch Size:** 16 (FP32)
* **Epochs Trained:** 20 epochs (421.95s runtime)
* **Model Selection Criterion:** Validation loss minimization (`EarlyStopping` monitored `val_loss`).
* **Selected Best Checkpoint:** Epoch 18 (`val_loss: 0.041658`, `val_acc: 93.90%`).
* **Saved Checkpoint:** `research_audit/checkpoints/retrained_best_model.pth` (100,934,975 bytes).

### Test Partition Blindness
* Inspection of `scripts/train_clean.py` lines 122–126 confirms that the test dataset was **never imported, instantiated, or accessed** during training or checkpoint selection.
* Test evaluation occurred strictly downstream via `scripts/evaluate_clean_model.py`.

---

## 4. Evaluation Metric Validation

The metrics stored in `research_audit/evaluation/clean_test_results.json` and documented in `research_audit/FINAL_MODEL_EVALUATION.md` were independently recalculated from the raw prediction confusion matrix:

$$\text{Confusion Matrix} = \begin{bmatrix} 62 & 1 & 0 \\ 3 & 15 & 0 \\ 0 & 0 & 84 \end{bmatrix} \begin{matrix} \text{Normal } (N=63) \\ \text{Benign } (N=18) \\ \text{Malignant } (N=84) \end{matrix}$$

### Mathematical Verification

1. **Overall Accuracy:**
   $$\text{Accuracy} = \frac{62 + 15 + 84}{165} = \frac{161}{165} \approx 0.975758 \quad (97.58\%)$$
   *Audit Result: Exact match with reported metric.*

2. **Per-Class Metrics:**
   * **Normal ($N=63$):**
     * $\text{Precision} = \frac{62}{62 + 3} = \frac{62}{65} \approx 0.953846 \quad (95.38\%)$
     * $\text{Recall} = \frac{62}{62 + 1} = \frac{62}{63} \approx 0.984127 \quad (98.41\%)$
     * $\text{F1-Score} = \frac{2 \times 62}{65 + 63} = \frac{124}{128} = 0.968750 \quad (96.88\%)$
     * $\text{Specificity} = \frac{99}{99 + 3} = \frac{99}{102} \approx 0.970588 \quad (97.06\%)$
   * **Benign ($N=18$):**
     * $\text{Precision} = \frac{15}{15 + 1} = \frac{15}{16} = 0.937500 \quad (93.75\%)$
     * $\text{Recall} = \frac{15}{15 + 3} = \frac{15}{18} \approx 0.833333 \quad (83.33\%)$
     * $\text{F1-Score} = \frac{2 \times 15}{16 + 18} = \frac{30}{34} \approx 0.882353 \quad (88.24\%)$
     * $\text{Specificity} = \frac{146}{146 + 1} = \frac{146}{147} \approx 0.993197 \quad (99.32\%)$
   * **Malignant ($N=84$):**
     * $\text{Precision} = \frac{84}{84 + 0} = 1.0000 \quad (100.00\%)$
     * $\text{Recall} = \frac{84}{84 + 0} = 1.0000 \quad (100.00\%)$
     * $\text{F1-Score} = 1.0000 \quad (100.00\%)$
     * $\text{Specificity} = \frac{81}{81 + 0} = 1.0000 \quad (100.00\%)$

3. **Macro Averages:**
   * $\text{Macro Precision} = \frac{0.953846 + 0.937500 + 1.000000}{3} \approx 0.963782 \quad (96.38\%)$
   * $\text{Macro Recall} = \frac{0.984127 + 0.833333 + 1.000000}{3} \approx 0.939153 \quad (93.92\%)$
   * $\text{Macro F1} = \frac{0.968750 + 0.882353 + 1.000000}{3} \approx 0.950368 \quad (95.04\%)$

4. **Weighted Averages:**
   * $\text{Weighted F1} = \frac{63 \times 0.968750 + 18 \times 0.882353 + 84 \times 1.000000}{165} \approx 0.975234 \quad (97.52\%)$

*Audit Result: All values in the JSON and Markdown report match theoretical derivations with zero discrepancy.*

---

## 5. Scientific Limitations & Boundaries of Claims

To maintain research rigor and ethical transparency, the following scientific boundaries must be observed in publications and documentation:

1. **Slice-Level vs. Patient-Level Independence:**
   * *Status:* The IQ-OTH/NCCD Kaggle distribution was de-identified by stripping all DICOM metadata, case IDs, and z-axis acquisition order.
   * *Requirement:* **Do not claim patient-level clinical validation**. While exact duplicate slices are quarantined and zero hash overlap is proven, multiple CT scan slices from the same original patient could theoretically exist across partitions if distinct slices from a single scan share anatomical features. This is a fundamental dataset constraint that must remain documented.
2. **Benign Class Sample Size:**
   * *Status:* The test set has 18 Benign slices (reflecting the 120 total benign slices in the entire archive).
   * *Requirement:* The 83.33% recall (15/18) has a binomial standard error of $\approx 8.8\%$. Small changes in single slice classifications cause $\pm 5.56\%$ shifts. This must be presented with sample support context ($N=18$).
3. **Malignant Generalization:**
   * *Status:* The model achieved 84/84 true positives on clean test slices.
   * *Requirement:* While impressive, this represents empirical performance on this benchmark subset, not a clinical proof of zero false negatives across arbitrary hospital scanners or out-of-distribution CT protocols.
4. **Input Modality:**
   * Standard 8-bit RGB $224 \times 224$ images rather than native 16-bit 3D volumetric DICOM slices with full Hounsfield Unit (-1000 to +1000 HU) dynamic range.

---

## 6. Regression Testing & Test Suite Audit

Executed: `python -m unittest discover -s tests -v`
* **Test Count:** 16 tests across 5 modules
* **Execution Time:** 16.39 seconds
* **Status:** **ALL 16 TESTS PASSED (0 FAILURES, 0 ERRORS)**

### Test Suite Inventory
* `tests/test_clean_splits.py`:
  * `test_manifests_exist`: Passed
  * `test_total_image_count_1097`: Passed (All 1,097 raw images accounted for)
  * `test_zero_hash_overlap`: Passed (Zero cross-partition SHA-256 collisions)
* `tests/test_model.py`:
  * `test_attention_module_shapes`: Passed (CBAM channel & spatial attention tensor shapes)
  * `test_checkpoint_compatibility`: Passed (Strict state dict loading)
  * `test_lung_cancer_model_forward`: Passed (Batch forward pass shape `(B, 3)`)
* `tests/test_preprocessing.py`:
  * `test_train_transform_shape`: Passed
  * `test_eval_transform_shape`: Passed
  * `test_unnormalize_tensor_bounds`: Passed (Clamping to `[0.0, 1.0]`)
* `tests/test_robustness.py`:
  * `test_fgsm_attack_bounds`: Passed (Valid perturbation range)
  * `test_input_validator_valid_image`: Passed
  * `test_input_validator_small_image`: Passed
  * `test_input_validator_degenerate_image`: Passed
  * `test_uncertainty_quantifier`: Passed (Monte Carlo Dropout variance computation)
* `tests/test_explainability.py`:
  * `test_gradcam_explainer`: Passed (Heatmap resolution and bounding)
  * `test_integrated_gradients_explainer`: Passed (Attribution convergence)

---

## 7. Git Working Tree & Artifact Inventory

* **Branch:** `main` (synced with `origin/main`)
* **Tracked Changes:** Only 2 functional bug fixes in working tree:
  * `src/evaluation/evaluator.py`: Added missing `Any` typing import.
  * `src/preprocessing/prepare_data.py`: Added idempotent directory wiping before split copying.
* **Historical Checkpoints Untouched:**
  * `checkpoints/best_model.pth` (SHA-256: `697158987784DEC839DAAA090FE8F5CD0273D3EF10C065F0F1022FA35F44CCF4`)
  * `checkpoints/final_model.pth` (SHA-256: `DE700CBAF4D57397904CF0153AF917370141E15F8409AC85B120E05B5FB15DFE`)
* **Historical Outputs Untouched:** `outputs/final_results.json`, `outputs/final_report.txt`, `outputs/metrics.json`.
* **New Audit Artifacts:** Isolated under `research_audit/`. No files staged, committed, or pushed to remote.

---

## 8. Final Audit Certification

$$\boxed{\textbf{AUDIT CERTIFICATION: PASSED (SCIENTIFICALLY RIGOROUS & REPRODUCIBLE)}}$$

The PulmoAI scientific correction is fully certified. The data leakage problem has been systematically investigated, quantified, and resolved. All reported clean metrics reflect genuine generalization on unseen test data, and all automated unit tests pass.
