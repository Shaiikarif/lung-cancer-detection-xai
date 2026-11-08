# Research Methodology: PulmoAI

## 1. Dataset Characteristics

The framework is developed and evaluated on the **IQ-OTH/NCCD Lung Cancer CT Dataset**, compiled by the Iraq-Oncology Teaching Hospital and the National Center for Cancer Diseases.

### Class Distribution:
- **Normal:** 416 axial slices from healthy individuals showing clear pulmonary parenchyma without focal lesions.
- **Benign:** 120 axial slices presenting benign granulomas, infectious infiltrates, or stable non-malignant nodules.
- **Malignant:** 561 axial slices exhibiting primary pulmonary adenocarcinoma, squamous cell carcinoma, or aggressive metastatic lesions.
- **Total Dataset Size:** 1,097 slices.

### Data Partitioning:
Stratified random sampling is conducted using seed `42`:
- **Training Set (70%):** 767 slices
- **Validation Set (15%):** 164 slices
- **Test Set (15%):** 289 slices (Normal: 122, Benign: 18, Malignant: 149)

---

## 2. Image Preprocessing & Augmentation Protocol

### Preprocessing:
1. **Geometric Standardization:** Bilinear interpolation resize to $224 \times 224 \times 3$.
2. **Contrast Enhancement:** Contrast Limited Adaptive Histogram Equalization (CLAHE) with clip limit 4.0.
3. **Statistical Z-Score Normalization:** Scaled using standard ImageNet channel moments:
   - Mean: $\mu = [0.485, 0.456, 0.406]$
   - Standard Deviation: $\sigma = [0.229, 0.224, 0.225]$

### Data Augmentation (Training Phase Only):
- Horizontal reflection ($p=0.5$)
- Vertical reflection ($p=0.3$)
- 90-degree orthogonal rotation ($p=0.3$)
- Spatial shift, scale, and continuous rotation ($\pm 15^\circ$, $p=0.5$)
- Brightness/contrast jitter ($\pm 20\%$, $p=0.4$)
- Gaussian spatial blur ($3 \times 3$ or $5 \times 5$, $p=0.2$)
- Coarse dropout / cutout simulation ($p=0.2$)

---

## 3. Training Dynamics & Optimization

- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$)
- **Initial Learning Rate:** $10^{-4}$
- **Scheduler:** Cosine Annealing with Warm Restarts ($T_0 = 10, T_{\text{mult}} = 2, \eta_{\text{min}} = 10^{-6}$)
- **Batch Size:** 32
- **Objective Function:** Multi-class Focal Loss with $\gamma = 2.0$
- **Regularization:** Dual dropout (0.5 and 0.25), Batch Normalization, and Early Stopping with 10-epoch patience based on validation loss.

---

## 4. Robustness Evaluation Methodology

### 4.1 FGSM Adversarial Testing
Adversarial vulnerability is systematically assessed using the Fast Gradient Sign Method:
$$\mathbf{x}_{\text{adv}} = \text{clip}_{0, 1}\left(\mathbf{x} + \epsilon \cdot \text{sign}\left(\nabla_{\mathbf{x}} \mathcal{L}(\theta, \mathbf{x}, y)\right)\right)$$
Perturbation budgets evaluated: $\epsilon \in \{0.0, 0.005, 0.01, 0.02, 0.03, 0.05\}$.

### 4.2 Corruption and Domain Perturbations
The test set is subjected to synthetic domain shifts simulating equipment degradation and scanning artifacts:
- Additive Gaussian Noise ($\sigma \in \{0.05, 0.10\}$)
- Spatial Blur ($k \in \{3, 5\}$)
- Illumination shifts ($\pm 20\%$)
- Radio-opaque occlusion masks ($10\%$ and $20\%$ spatial area)

### 4.3 Epistemic Uncertainty Estimation
Monte Carlo Dropout is enabled at test time. For each input image $\mathbf{x}$, $T=30$ forward passes are drawn:
$$\mu_c = \frac{1}{T} \sum_{t=1}^T p_c^{(t)}, \quad \sigma_c = \sqrt{\frac{1}{T} \sum_{t=1}^T \left(p_c^{(t)} - \mu_c\right)^2}$$
$$H(p) = -\sum_{c=1}^C \mu_c \log(\mu_c + \epsilon)$$
Predictions with variance $\sigma > 0.15$ or confidence $\mu < 0.70$ are flagged as clinically ambiguous.
