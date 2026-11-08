# Limitations and Clinical Disclaimers

## 1. Scientific & Engineering Limitations

### 1.1 2D Slice Representation vs. 3D Volumetric Context
- **Current State:** The model classifies isolated 2D axial CT slices.
- **Clinical Reality:** In radiology, pulmonary nodules and masses are three-dimensional anatomical structures. A single 2D slice may cut through the periphery of a lesion or capture normal vascular bifurcation mimicking a nodule. Full volumetric 3D analysis (e.g. using 3D CNNs or Vision Transformers over complete volumetric DICOM series) is necessary for full clinical lesion staging.

### 1.2 Dataset Size and Distributional Skew
- **Cohort Size:** Evaluated on the IQ-OTH/NCCD dataset containing 1,097 slices from 110 patient cases. While sufficient for proof-of-concept transfer learning, multi-center generalization requires external validation on larger, diverse benchmarks such as LIDC-IDRI or NLST.
- **Benign Class Under-representation:** The benign cohort comprises only 18 test slices, resulting in wider confidence intervals for benign recall (88.89%) relative to malignant recall (100.0%).

### 1.3 Format & Metadata Constraints
- The dataset utilizes pre-windowed standard image formats (JPEG/PNG) rather than raw DICOM files with calibrated Hounsfield Units (HU). Calibrated HU windowing (e.g. lung window: $W=1500, L=-600$; mediastinal window: $W=350, L=50$) preserves radiodensity information lost in 8-bit conversions.

### 1.4 Adversarial Vulnerability
- Although the framework quantifies robustness against FGSM perturbations, adversarial defense (e.g. adversarial training or randomized smoothing) is not currently enabled, meaning deliberate or out-of-distribution adversarial noise can degrade model accuracy.

---

## 2. Regulatory & Medical Disclaimer

> [!WARNING]
> **RESEARCH & DEMONSTRATION USE ONLY**
>
> PulmoAI is an academic research software prototype and **IS NOT** an FDA/CE-cleared medical device or diagnostic system.
>
> - **DO NOT** use this software to diagnose, treat, cure, or prevent any medical condition.
> - **DO NOT** base clinical patient management decisions on outputs produced by this software.
> - All outputs, visual explanations, and uncertainty flags must be interpreted by a certified radiologist, oncologist, or pulmonologist.
> - The developers and contributors assume no liability for errors, misdiagnoses, or omissions resulting from the application of this code.
