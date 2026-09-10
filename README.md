# Motion-Guided Self-Supervised Learning for Label-Efficient Cardiac Cine MRI Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents
- [A. Project Overview](#a-project-overview)
- [B. Dataset Acquisition](#b-dataset-acquisition)
- [C. Dataset Structure](#c-dataset-structure)
- [D. Preprocessing](#d-preprocessing)
- [E. Patient-Level Splitting](#e-patient-level-splitting)
- [F. Baseline Cardiac Segmentation Model](#f-baseline-cardiac-segmentation-model)
- [G. Self-Supervised Temporal Pretraining (SSL)](#g-self-supervised-temporal-pretraining-ssl)
- [H. Motion Estimation & Temporal Consistency](#h-motion-estimation--temporal-consistency)
- [I. Confidence-Filtered Pseudo-Labeling](#i-confidence-filtered-pseudo-labeling)
- [J. Limited-Label Fine-Tuning & Label Efficiency](#j-limited-label-fine-tuning--label-efficiency)
- [K. Evaluation Protocol & Multi-Level Metrics](#k-evaluation-protocol--multi-level-metrics)
- [L. Component Ablation Study](#l-component-ablation-study)
- [M. Robustness & Stability Analysis](#m-robustness--stability-analysis)
- [N. Training on a Separate GPU Machine](#n-training-on-a-separate-gpu-machine)
- [O. Reproducibility & Environment Setup](#o-reproducibility--environment-setup)

---

## A. Project Overview

This project implements a motion-guided self-supervised learning (SSL) framework for label-efficient cardiac cine MRI segmentation using the MICCAI Automated Cardiac Diagnosis Challenge (ACDC) dataset.

### Core Scientific Hypothesis
Cardiac cine MRI datasets contain dense temporal sequences capturing continuous biomechanical heart motion across the cardiac cycle (typically 20–35 temporal frames per slice). However, clinical ground truth segmentations are restricted exclusively to End-Diastole (ED) and End-Systole (ES). By exploiting unlabeled intermediate cine frames through self-supervised temporal representation learning, differentiable motion estimation, and confidence-filtered pseudo-labeling, we can substantially improve 4-class segmentation performance in extreme low-label regimes ($10\%$, $25\%$, and $50\%$ labeled patients).

### 4-Class Segmentation Task
- **Class 0**: Background
- **Class 1**: Left Ventricle (LV) Cavity
- **Class 2**: Myocardium (MYO)
- **Class 3**: Right Ventricle (RV) Cavity

---

## B. Dataset Acquisition

The project strictly uses the official **Automated Cardiac Diagnosis Challenge (ACDC)** dataset:
- **Official Webpage**: [CREATIS ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
- **Official Data Repository**: [Human Heart Project (INSA Lyon)](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)
- **Official REST Endpoint**: `https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/63721d7073e9f0047faa0525/download`

### Automated Acquisition:
```bash
python src/download_acdc.py --dest data/raw/ACDC
```
The script performs automatic download, SHA-256 integrity verification, and folder extraction into `data/raw/ACDC/training/` (100 patients, `patient001`–`patient100`).

---

## C. Dataset Structure

```
motion_guided_ssl_project/
├── configs/                       # Parameterized YAML configurations
│   ├── base_config.yaml           # Central experiment hyperparameters
│   ├── baseline_config.yaml       # Supervised baseline 2D U-Net config
│   ├── ssl.yaml                   # Temporal masked reconstruction config
│   ├── motion.yaml                # Differentiable flow and warping config
│   ├── pseudo_labels.yaml         # Confidence filtering & calibration config
│   ├── experiments.yaml           # Ablation variants & label regimes config
│   └── preprocessing_config.yaml  # Resampling, cropping, normalization config
├── data/
│   ├── raw/ACDC/                  # Original ACDC NIfTI volumes
│   ├── processed/                 # Standardized 2D slices (.npz compressed)
│   └── splits/                    # Fixed patient-level split files
│       ├── train_patients.txt     # 70 patients (70%)
│       ├── val_patients.txt       # 10 patients (10%)
│       ├── test_patients.txt      # 20 patients (20%)
│       ├── labeled_10.txt         #  7 patients (10% label regime)
│       ├── labeled_25.txt         # 17 patients (25% label regime)
│       ├── labeled_50.txt         # 35 patients (50% label regime)
│       └── labeled_100.txt        # 70 patients (100% label regime)
├── notebooks/                     # Interactive Jupyter walkthroughs
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_segmentation.ipynb
│   ├── 04_self_supervised_pretraining.ipynb
│   ├── 05_motion_consistency.ipynb
│   ├── 06_confidence_pseudo_labels.ipynb
│   ├── 07_limited_label_finetuning.ipynb
│   └── 08_ablation_robustness.ipynb
├── results/                       # Outputs, comparison tables & figures
│   ├── experiments/               # Experiment registry JSON & training logs
│   ├── tables/                    # Ablation, label-efficiency & patient tables
│   └── figures/                   # Publication-quality diagnostic plots
├── scripts/
│   └── smoke_test.py              # Unified 10-step end-to-end CPU smoke test
├── src/                           # Core implementation modules
│   ├── dataset.py                 # PyTorch datasets with patient management
│   ├── encoder.py                 # 4-stage residual SharedEncoder
│   ├── segmentation_model.py     # 2D U-Net segmentation architecture
│   ├── ssl.py                     # Temporal masked autoencoder & losses
│   ├── motion.py                  # FlowNet & SpatialTransformer warping
│   ├── pseudo_labels.py           # Selective confidence filtering engine
│   ├── losses.py                  # DiceCELoss & consistency formulations
│   ├── metrics.py                 # Patient-level Dice, HD95 & temporal metrics
│   ├── experiment_runner.py       # Fine-tuning & ablation runner
│   └── aggregate_results.py       # Table aggregation & plot generators
├── requirements.txt               # Reproducible pip dependencies
├── environment.yml                # Conda/mamba environment specification
└── README.md
```

---

## D. Preprocessing

Standardized spatial and intensity preprocessing is implemented in [`src/preprocess.py`](file:///c:/Users/Saket/OneDrive/Desktop/UROP/motion_guided_ssl_project/src/preprocess.py):
1. **Spatial Resampling**: In-plane physical spacing resampled to isotropic $1.5 \times 1.5\text{ mm}$ using order-3 spline interpolation for images and order-0 nearest-neighbor for segmentation masks.
2. **Fixed Dimension Padding/Cropping**: Centered spatial window of $256 \times 256$ pixels.
3. **Per-Volume Intensity Normalization**: Non-zero cardiac voxel $z$-score standardization ($(\mu, \sigma)$ calculated across the 4D volume).
4. **Label Remapping**: Official ACDC labels ($0=\text{BG}, 1=\text{RV}, 2=\text{MYO}, 3=\text{LV}$) remapped to project specification ($0=\text{BG}, 1=\text{LV}, 2=\text{MYO}, 3=\text{RV}$).
5. **Output Format**: Compressed 2D `.npz` files containing `image`, `mask` ($-1$ for intermediate cine frames), `patient_id`, `slice_idx`, `frame_idx`, and `phase`.

```bash
# Execute preprocessing on raw ACDC data
python src/preprocess.py --config configs/preprocessing_config.yaml
```

---

## E. Patient-Level Splitting

Splits are strictly partitioned at the **patient level** to prevent spatial data leakage across slices:
- **Training Cohort**: 70 patients (1,324 labeled ED/ES slices, 22,000+ intermediate cine slices)
- **Validation Cohort**: 10 patients (194 labeled slices)
- **Test Cohort**: 20 patients (386 labeled slices)
- **Pathology Stratification**: Balanced distribution across all 5 diagnostic groups (NOR, MINF, DCM, HCM, ARV).

### Nested Labeled Subsets (Seed 42):
$$\text{Patients}_{10\%} (7\text{ pts}) \subset \text{Patients}_{25\%} (17\text{ pts}) \subset \text{Patients}_{50\%} (35\text{ pts}) \subset \text{Patients}_{100\%} (70\text{ pts})$$
Validation and test patients have **zero overlap** with training subsets.

---

## F. Baseline Cardiac Segmentation Model

Implemented in [`src/segmentation_model.py`](file:///c:/Users/Saket/OneDrive/Desktop/UROP/motion_guided_ssl_project/src/segmentation_model.py):
- **Architecture**: 2D U-Net with 4-stage residual convolutional encoder (`[32, 64, 128, 256]` channels) and symmetric decoder with skip connections.
- **Parameters**: 2,012,580 trainable parameters.
- **Loss Function**: Compound Dice + Cross-Entropy Loss:
  $$\mathcal{L}_{\text{sup}} = \mathcal{L}_{\text{dice}} + \mathcal{L}_{\text{ce}}$$
  Background class 0 is excluded from the Dice calculation to focus gradient flow on small cardiac structures.

---

## G. Self-Supervised Temporal Pretraining (SSL)

Implemented in [`src/ssl.py`](file:///c:/Users/Saket/OneDrive/Desktop/UROP/motion_guided_ssl_project/src/ssl.py):
- **Objective**: Learn anatomical and dynamic cardiac representations from unlabeled cine frames before segmentation fine-tuning.
- **Components**:
  1. *Patch Masking*: Divides input frame $I_t$ into $16 \times 16$ non-overlapping patches and masks a random $50\%$ of them.
  2. *Masked Reconstruction*: Lightweight convolutional decoder reconstructs unmasked pixel intensities ($L_1$ error).
  3. *Temporal Feature Consistency*: Projection MLP enforces cosine consistency between clean encoder representations of adjacent cine frames from the same sequence:
     $$\mathcal{L}_{\text{ssl}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{temporal}} \mathcal{L}_{\text{temporal}}$$

---

## H. Motion Estimation & Temporal Consistency

Implemented in [`src/motion.py`](file:///c:/Users/Saket/OneDrive/Desktop/UROP/motion_guided_ssl_project/src/motion.py):
- **Motion Estimator**: Lightweight convolutional network (`SimpleFlowNet`, 168k parameters) predicting dense pixel displacement field $\mathbf{u}_{t \to t+1} \in \mathbb{R}^{2 \times H \times W}$.
- **Spatial Transformer**: Differentiable bilinear grid sampling warping frame $t$ to frame $t+1$:
  $$\hat{I}_{t+1} = \mathcal{W}(I_t, \mathbf{u}_{t \to t+1})$$
- **Loss Formulation**:
  $$\mathcal{L}_{\text{motion}} = \|\hat{I}_{t+1} - I_{t+1}\|_1 + \lambda_{\text{smooth}} \text{TV}(\mathbf{u})$$
- Enables motion-warped mask propagation and temporal regularization during fine-tuning.

---

## I. Confidence-Filtered Pseudo-Labeling

Implemented in [`src/pseudo_labels.py`](file:///c:/Users/Saket/OneDrive/Desktop/UROP/motion_guided_ssl_project/src/pseudo_labels.py):
- **Selective Filtering**: Never assigns pseudo-labels indiscriminately.
- **Confidence Metrics**:
  - Maximum Softmax Probability (MSP): $C(x) = \max_c P(y=c|x)$.
  - Normalized Shannon Entropy: $H(x) = -\sum_c P_c \log P_c / \log C$.
  - Motion-Warped Agreement: Agreement with warped adjacent prediction.
- **Acceptance Rule**: Pixels satisfying $C(x) \ge \tau$ (default $\tau = 0.90$) are assigned hard pseudo-labels; uncertain pixels are set to `ignore_index = -1` and omitted from loss backpropagation.

---

## J. Limited-Label Fine-Tuning & Label Efficiency

Implemented in [`src/experiment_runner.py`](file:///c:/Users/Saket/OneDrive/Desktop/UROP/motion_guided_ssl_project/src/experiment_runner.py):
- Evaluates the **16-experiment matrix** (4 label fractions $\times$ 4 model variants).
- Multi-objective joint loss formulation:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sup}} + \lambda_{\text{motion}} \mathcal{L}_{\text{motion}} + \lambda_{\text{pseudo}} \mathcal{L}_{\text{pseudo}}$$
- **Differential Learning Rates**: Encoder initialized with SSL weights trained at $1.0 \times 10^{-5}$; decoder trained at $1.0 \times 10^{-4}$.

---

## K. Evaluation Protocol & Multi-Level Metrics

Implemented in [`src/metrics.py`](file:///c:/Users/Saket/OneDrive/Desktop/UROP/motion_guided_ssl_project/src/metrics.py):
1. **Patient-Level Metrics**: Metrics are computed per patient first, then averaged across the 20 test patients ($\text{mean} \pm \text{std}$) to prevent slice count bias.
   - Per-class Dice: LV, Myocardium, RV, Mean Foreground Dice.
   - 95th Percentile Hausdorff Distance (HD95 in mm).
2. **Temporal Consistency Metrics**:
   - Frame-to-frame uncompensated agreement and Dice.
   - Motion-compensated agreement and Dice using spatial transformer warping.
   - Motion compensation gain: $\Delta_{\text{motion}} = \text{Dice}_{\text{warped}} - \text{Dice}_{\text{raw}}$.
3. **Pseudo-Label Calibration**:
   - Pixel acceptance rates across confidence thresholds ($\tau \in [0.60, 0.95]$).
   - Precision, accuracy, and per-class Dice on accepted pixels.

---

## L. Component Ablation Study

Configured in [`configs/experiments.yaml`](file:///c:/Users/Saket/OneDrive/Desktop/UROP/motion_guided_ssl_project/configs/experiments.yaml):

| Variant Key | Display Name | Pretrained SSL | Motion Regularization | Pseudo-Labels | Objective |
|---|---|:---:|:---:|:---:|---|
| `supervised` | **A. Supervised Baseline** | — | — | — | Randomly initialized 2D U-Net |
| `ssl_finetune` | **B. SSL Only** | $\checkmark$ | — | — | SSL-pretrained encoder fine-tuning |
| `ssl_motion` | **C. SSL + Motion** | $\checkmark$ | $\checkmark$ | — | SSL fine-tuning + motion regularization |
| `ssl_pseudo` | **D. SSL + Pseudo-Labels** | $\checkmark$ | — | $\checkmark$ | SSL fine-tuning + confidence pseudo-labels |
| `full_pipeline` | **E. Full Proposed Pipeline** | $\checkmark$ | $\checkmark$ | $\checkmark$ | Integrated SSL + Motion + Pseudo-Labels |

Comparison tables generated automatically via [`src/aggregate_results.py`](file:///c:/Users/Saket/OneDrive/Desktop/UROP/motion_guided_ssl_project/src/aggregate_results.py) to `results/tables/ablation_table.csv`.

---

## M. Robustness & Stability Analysis

Evaluates performance stability under controlled perturbations:
- **Multi-Seed Stability**: Identical configurations trained with seeds `42`, `123`, `456`.
- **Additive Gaussian Noise**: $\sigma \in [0.0, 0.05, 0.10]$ applied at test time.
- **Global Intensity Scaling**: Multiplicative factors $0.9\times, 1.0\times, 1.1\times$.
- **Temporal Interval Spacing**: Cine temporal stride $\Delta t = 1$ vs. $\Delta t = 2$.

---

## N. Training on a Separate GPU Machine

All expensive model training must be executed on the dedicated GPU training cluster.

### Step 1: Environment Setup
```bash
git clone <repo-url> motion_guided_ssl_project
cd motion_guided_ssl_project

# Option A: Conda
conda env create -f environment.yml
conda activate cardiac_ssl

# Option B: Pip / venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Step 2: Acquire & Preprocess Dataset
```bash
# Download raw ACDC dataset
python src/download_acdc.py --dest data/raw/ACDC

# Standardize 2D slices
python src/preprocess.py --config configs/preprocessing_config.yaml
```

### Step 3: Train Baseline 2D U-Net (RUN ON TRAINING MACHINE)
```bash
python src/train.py --config configs/baseline_config.yaml --device cuda
```

### Step 4: Self-Supervised Temporal Pretraining (RUN ON TRAINING MACHINE)
```bash
python src/ssl.py --config configs/ssl.yaml --device cuda
```

### Step 5: Motion Estimator Pretraining (RUN ON TRAINING MACHINE)
```bash
python src/motion.py --config configs/motion.yaml --device cuda
```

### Step 6: Generate Filtered Pseudo-Labels (RUN ON TRAINING MACHINE)
```bash
python src/pseudo_labels.py --config configs/pseudo_labels.yaml --device cuda
```

### Step 7: Train the 5 Ablation Variants (10% Labels) (RUN ON TRAINING MACHINE)
```bash
python src/experiment_runner.py --mode supervised --label-fraction 10 --device cuda
python src/experiment_runner.py --mode ssl_finetune --label-fraction 10 --device cuda
python src/experiment_runner.py --mode ssl_motion --label-fraction 10 --device cuda
python src/experiment_runner.py --mode ssl_pseudo --label-fraction 10 --device cuda
python src/experiment_runner.py --mode full_pipeline --label-fraction 10 --device cuda
```

### Step 8: Train Label-Efficiency Matrix (25%, 50%, 100%) (RUN ON TRAINING MACHINE)
```bash
for frac in 25 50 100; do
    python src/experiment_runner.py --mode supervised --label-fraction $frac --device cuda
    python src/experiment_runner.py --mode ssl_finetune --label-fraction $frac --device cuda
    python src/experiment_runner.py --mode ssl_motion --label-fraction $frac --device cuda
    python src/experiment_runner.py --mode full_pipeline --label-fraction $frac --device cuda
done
```

### Step 9: Multi-Seed Robustness Experiments (RUN ON TRAINING MACHINE)
```bash
python src/experiment_runner.py --mode full_pipeline --label-fraction 10 --seed 123 --device cuda
python src/experiment_runner.py --mode full_pipeline --label-fraction 10 --seed 456 --device cuda
```

### Step 10: Generate Comparison Tables & Figures
```bash
python src/aggregate_results.py --generate-plots
```

---

## O. Reproducibility & Environment Setup

- **Deterministic Seeding**: Global seeds set across Python `random`, `numpy`, and `torch` (CUDA deterministic mode enabled).
- **Patient Isolation**: Fixed patient split files guarantee zero data leakage between training, validation, and testing.
- **Machine-Independent Execution**: All dataset and checkpoint paths are relative or configurable via CLI.
- **Lightweight CPU Smoke Test**: Verify system integrity without GPU resources in under 15 seconds:
  ```bash
  python scripts/smoke_test.py
  ```
  Result: **10/10 pipeline checks passed successfully**.
"# Self-Supervised-Learning-for-Label-Efficient-Cardiac-Cine-MRI-Analysis" 
