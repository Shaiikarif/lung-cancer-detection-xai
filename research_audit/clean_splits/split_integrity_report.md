# Clean Dataset Split & Partition Integrity Report

## 1. Split Specification

- **Source Dataset:** Raw `The IQ-OTHNCCD lung cancer dataset`
- **Partition Ratios:** 70% Train / 15% Validation / 15% Test (Stratified)
- **Deduplication Strategy:** Hash clustering. Identical SHA-256 hashes are placed in the same partition.
- **Random Seed:** `42`
- **Total Images:** `1097`
- **Total Unique Hashes:** `1054`

## 2. Partition Summary Table

| Partition | Benign | Malignant | Normal | Total Images | Total Hashes | Percentage |
|---|---|---|---|---|---|---|
| **Train** | 84 | 393 | 291 | 768 | 743 | 70.01% |
| **Val** | 18 | 84 | 62 | 164 | 155 | 14.95% |
| **Test** | 18 | 84 | 63 | 165 | 156 | 15.04% |
| **Total** | **120** | **561** | **416** | **1097** | **1054** | **100.00%** |

## 3. Cross-Partition Overlap Verification

| Overlap Verification Pair | Shared SHA-256 Hashes | Shared Files | Status |
|---|---|---|---|
| Train vs. Validation | `0` | `0` | PASS (Zero Leakage) |
| Train vs. Test | `0` | `0` | PASS (Zero Leakage) |
| Validation vs. Test | `0` | `0` | PASS (Zero Leakage) |

## 4. Manifest Artifacts

- `research_audit/clean_splits/train_manifest.csv`
- `research_audit/clean_splits/validation_manifest.csv`
- `research_audit/clean_splits/test_manifest.csv`
- `research_audit/clean_splits/split_summary.json`
- Clean data folder: `research_audit/clean_splits/data/{train,val,test}/{Benign,Malignant,Normal}`
