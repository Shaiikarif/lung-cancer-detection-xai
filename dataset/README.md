# Dataset Directory

This directory stores the partitioned CT scan dataset for lung cancer detection.

## Dataset Information:
- **Source:** [The IQ-OTH/NCCD Lung Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)
- **Institution:** Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases (IQ-OTH/NCCD).
- **Format:** Axial Chest CT slices in JPG / PNG format.
- **Classes:**
  - `normal/`: Non-tumoral pulmonary tissue (416 images)
  - `benign/`: Non-cancerous nodules or inflammatory lesions (120 images)
  - `malignant/`: Primary pulmonary adenocarcinoma / carcinoma lesions (561 images)
  - **Total:** 1,097 slices

## Directory Structure:
```
dataset/
├── train/
│   ├── normal/
│   ├── benign/
│   └── malignant/
├── val/
│   ├── normal/
│   ├── benign/
│   └── malignant/
├── test/
│   ├── normal/
│   ├── benign/
│   └── malignant/
└── lung_cancer_symptoms.csv
```

## Setup Instructions:
1. Download the raw dataset from Kaggle or the study repository.
2. Place the unzipped folder `The IQ-OTHNCCD lung cancer dataset` in the repository root.
3. Run the automated data splitting script:
   ```bash
   python -m src.preprocessing.prepare_data
   ```
   This performs a stratified 70% / 15% / 15% split into `dataset/train/`, `dataset/val/`, and `dataset/test/`.
