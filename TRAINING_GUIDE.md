# Training Guide & Operational Handoff Package
## Motion-Guided Self-Supervised Learning for Cardiac Cine MRI

This guide provides complete operational instructions for transferring the codebase from the local development system to a remote GPU-enabled training server and executing all training stages without requiring code modifications.

---

## 1. Hardware & System Requirements

The following requirements are derived directly from the model architecture, tensor shapes, and loss formulations:

### Compute & GPU Memory (VRAM)
- **Minimum GPU Requirement**: NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (e.g., RTX 2060, RTX 3060, Tesla T4).
- **Recommended GPU**: NVIDIA GPU with $\ge 8–16\text{ GB}$ VRAM (e.g., RTX 3080/4080, V100, A10, A100).
- **VRAM Breakdown by Stage**:
  - *Baseline 2D U-Net* (Batch size 8, $256 \times 256$, 4 classes): Activations $\approx 1.2\text{ GB}$, Weights $+$ Optimizer $\approx 80\text{ MB}$, CUDA runtime overhead $\approx 1.0\text{ GB}$. Total peak: $\approx 2.5–3.5\text{ GB}$.
  - *SSL Temporal Pretraining* (Batch size 16, $256 \times 256$ paired frames): Dual encoder forward pass $+$ projection head $+$ patch reconstruction decoder. Total peak: $\approx 4.0–5.0\text{ GB}$.
  - *Motion Estimator Pretraining* (Batch size 16, concatenated frame pairs): SimpleFlowNet $+$ grid sampling warping. Total peak: $\approx 3.5–4.5\text{ GB}$.
  - *Full Pipeline Fine-Tuning* (Batch size 8, segmentation $+$ motion loss $+$ pseudo-label loss): Total peak: $\approx 4.5–6.0\text{ GB}$.
  *(Note: If training on a 4 GB GPU, reduce batch sizes by 50% in the respective YAML config file.)*

### Software & Drivers
- **Operating System**: Linux (Ubuntu 20.04/22.04 LTS recommended) or Windows 10/11 with NVIDIA drivers.
- **CUDA Version**: CUDA 11.8 or CUDA 12.1+ (compatible with PyTorch 2.0+).
- **Python Version**: Python 3.9, 3.10, or 3.11 (tested on Python 3.11).
- **PyTorch**: `torch >= 2.0.0`, `torchvision >= 0.15.0`.
- **MONAI**: `monai >= 1.3.0`.

### Disk Storage
- **Raw ACDC Dataset**: $\approx 2.5\text{ GB}$ (compressed zip $+$ extracted NIfTI volumes).
- **Preprocessed Slices**: $\approx 3.5\text{ GB}$ ($25,351$ compressed 2D `.npz` slices).
- **Model Checkpoints**: $\approx 500\text{ MB}$ (baseline, SSL, motion, and 16 experiment checkpoints).
- **Logs, Tables, & Figures**: $\approx 200\text{ MB}$.
- **Total Recommended Free Disk Space**: **$\ge 15\text{ GB}$** on SSD storage.

---

## 2. Environment Setup on Training Machine

### Option A: Conda / Mamba Environment (Recommended)
```bash
# Clone or copy repository to training server
git clone <repo-url> motion_guided_ssl_project
cd motion_guided_ssl_project

# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate cardiac_ssl
```

### Option B: Standard Python Virtualenv & Pip
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Quick Verification
Confirm environment readiness in under 15 seconds (runs entirely on CPU):
```bash
python scripts/smoke_test.py
```
Expected output: **`AUDIT & SMOKE TEST PASSED (10/10 CHECKS SUCCESSFUL)`**.

---

## 3. Dataset Setup

The code supports configurable dataset paths via YAML files and CLI arguments.

### Expected Directory Structure
Place the official ACDC challenge dataset in `data/raw/ACDC/`:
```
data/raw/ACDC/
└── training/
    ├── patient001/
    │   ├── Info.cfg
    │   ├── patient001_4d.nii.gz
    │   ├── patient001_frame01.nii.gz
    │   ├── patient001_frame01_gt.nii.gz
    │   ├── patient001_frame12.nii.gz
    │   └── patient001_frame12_gt.nii.gz
    ├── patient002/
    └── ... (patient001 through patient100)
```

### Automated Acquisition (If Direct Internet Access Available)
```bash
python src/download_acdc.py --dest data/raw/ACDC
```

### Manual Transfer (If Behind Cluster Firewall)
If downloading directly is not possible, copy the extracted `training/` folder from institutional storage to:
`<project_root>/data/raw/ACDC/training/`

Verify data integrity:
```bash
python src/dataset_inspector.py --data-dir data/raw/ACDC
```

---

## 4. Sequential Training Execution Order

All training must follow this strict 9-step dependency sequence:

```
[STEP 1] Preprocess Raw ACDC Dataset (Generates 25,351 .npz slices)
   │
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
[STEP 2] Baseline 2D U-Net    [STEP 3] SSL Pretraining       [STEP 4] Motion Pretraining
(Train on 100% labels)        (Train on adjacent frames)     (Train SimpleFlowNet)
   │                              │                              │
   │  ┌───────────────────────────┘                              │
   │  ▼                                                          ▼
[STEP 5] Pseudo-Labeling      [STEP 6] Limited-Label Matrix  [STEP 7] Multi-Level Eval
(MSP + Entropy filtering)     (10%, 25%, 50%, 100% fine-tune)(Patient Dice, HD95)
   │                              │                              │
   └──────────────────────────────┴──────────────────────────────┘
                                  │
                                  ▼
                     [STEP 8] Ablation Study (Variants A-E)
                                  │
                                  ▼
                     [STEP 9] Robustness & Plot Aggregation
```

---

## 5. Training Commands Reference

Commands are separated into **Development / CPU Smoke Tests** and **Training Machine GPU Commands**.

> [!IMPORTANT]
> **RUN EXPENSIVE COMMANDS ONLY ON THE GPU TRAINING MACHINE.**

### STEP 1: Preprocessing & Split Verification
*Processes raw NIfTI volumes into standardized $256 \times 256$ slices at $1.5\text{ mm}$ resolution.*
```bash
# TRAINING MACHINE COMMAND:
python src/preprocess.py --config configs/preprocessing_config.yaml
```

### STEP 2: Supervised Baseline 2D U-Net Training
*Establishes the upper-bound baseline using all 1,324 labeled slices from 70 training patients.*
```bash
# DEVELOPMENT / CPU TEST:
python src/train.py --config configs/baseline_config.yaml --device cpu

# TRAINING MACHINE COMMAND:
python src/train.py --config configs/baseline_config.yaml --device cuda
```
- **Output Checkpoint**: `checkpoints/baseline_unet_best.pth`

### STEP 3: Self-Supervised Temporal Pretraining (SSL)
*Pretrains SharedEncoder on adjacent temporal cine pairs using masked patch reconstruction and feature consistency.*
```bash
# DEVELOPMENT / CPU TEST:
python src/ssl.py --config configs/ssl.yaml --device cpu

# TRAINING MACHINE COMMAND:
python src/ssl.py --config configs/ssl.yaml --device cuda
```
- **Output Checkpoint**: `checkpoints/ssl/ssl_encoder_best.pth`

### STEP 4: Motion Estimator Pretraining
*Pretrains SimpleFlowNet for unsupervised dense cardiac motion estimation.*
```bash
# DEVELOPMENT / CPU TEST:
python src/motion.py --config configs/motion.yaml --device cpu

# TRAINING MACHINE COMMAND:
python src/motion.py --config configs/motion.yaml --device cuda
```
- **Output Checkpoint**: `checkpoints/motion/motion_model_best.pth`

### STEP 5: Confidence-Filtered Pseudo-Label Generation
*Generates high-confidence pseudo-segmentations on intermediate unlabeled cine frames.*
```bash
# DEVELOPMENT / CPU TEST:
python src/pseudo_labels.py --config configs/pseudo_labels.yaml --device cpu

# TRAINING MACHINE COMMAND:
python src/pseudo_labels.py --config configs/pseudo_labels.yaml --device cuda
```
- **Output Directory**: `results/pseudo_labels/`

### STEP 6: Limited-Label Fine-Tuning Matrix
*Fine-tunes the models under 10%, 25%, 50%, and 100% labeled patient regimes.*
```bash
# ========================================================
# TRAINING MACHINE COMMANDS (GPU REQUIRED)
# ========================================================

# Full Proposed Framework across all 4 label fractions
python src/experiment_runner.py --mode full_pipeline --label-fraction 10 --device cuda
python src/experiment_runner.py --mode full_pipeline --label-fraction 25 --device cuda
python src/experiment_runner.py --mode full_pipeline --label-fraction 50 --device cuda
python src/experiment_runner.py --mode full_pipeline --label-fraction 100 --device cuda
```

### STEP 7: Five-Variant Component Ablation Study (10% Labels)
*Evaluates the contribution of each modular framework component under extreme label scarcity.*
```bash
# ========================================================
# TRAINING MACHINE COMMANDS (GPU REQUIRED)
# ========================================================

# Variant A: Supervised Baseline (random init)
python src/experiment_runner.py --mode supervised --label-fraction 10 --device cuda

# Variant B: SSL Only (pretrained encoder, no motion, no pseudo)
python src/experiment_runner.py --mode ssl_finetune --label-fraction 10 --device cuda

# Variant C: SSL + Motion Regularization (no pseudo)
python src/experiment_runner.py --mode ssl_motion --label-fraction 10 --device cuda

# Variant D: SSL + Pseudo-Labels (no motion)
python src/experiment_runner.py --mode ssl_pseudo --label-fraction 10 --device cuda

# Variant E: Full Pipeline (SSL + Motion + Pseudo-Labels)
python src/experiment_runner.py --mode full_pipeline --label-fraction 10 --device cuda
```

### STEP 8: Multi-Seed Robustness Experiments
*Evaluates reproducibility across distinct random seeds.*
```bash
# ========================================================
# TRAINING MACHINE COMMANDS (GPU REQUIRED)
# ========================================================
python src/experiment_runner.py --mode full_pipeline --label-fraction 10 --seed 123 --device cuda
python src/experiment_runner.py --mode full_pipeline --label-fraction 10 --seed 456 --device cuda
```

### STEP 9: Generate Comparison Tables & Publication Figures
*Compiles all experiment registry outputs into Markdown/CSV tables and figures.*
```bash
python src/aggregate_results.py --generate-plots
```

---

## 6. Checkpoint Dependency & Flow

The following table details how model weights flow across training stages:

| Stage | Producer Script | Output Checkpoint | Consumer Script | Consumer Argument |
|---|---|---|---|---|
| **SSL Pretraining** | `src/ssl.py` | `checkpoints/ssl/ssl_encoder_best.pth` | `src/experiment_runner.py` | `--ssl-checkpoint` or `configs/experiments.yaml` |
| **Motion Pretraining** | `src/motion.py` | `checkpoints/motion/motion_model_best.pth` | `src/experiment_runner.py` | `configs/experiments.yaml:motion_checkpoint` |
| **Baseline Segmentation** | `src/train.py` | `checkpoints/baseline_unet_best.pth` | `src/pseudo_labels.py` | `configs/pseudo_labels.yaml:checkpoint` |
| **Fine-Tuning Runs** | `src/experiment_runner.py` | `checkpoints/experiments/{mode}_{frac}pct_best.pth` | `src/metrics.py` / `src/aggregate_results.py` | Registry tracking |

---

## 7. Experiment Matrix

| # | Experiment ID | Method / Variant | Label Fraction | Patient Count | Pretrained SSL? | Motion Regularization? | Pseudo-Labels? | Run on GPU? |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `supervised_10pct` | Supervised Baseline | 10% | 7 pts | No | No | No | **YES** |
| 2 | `supervised_25pct` | Supervised Baseline | 25% | 17 pts | No | No | No | **YES** |
| 3 | `supervised_50pct` | Supervised Baseline | 50% | 35 pts | No | No | No | **YES** |
| 4 | `supervised_100pct` | Supervised Baseline | 100% | 70 pts | No | No | No | **YES** |
| 5 | `ssl_finetune_10pct` | SSL Only (Ablation B) | 10% | 7 pts | **Yes** | No | No | **YES** |
| 6 | `ssl_finetune_25pct` | SSL Only | 25% | 17 pts | **Yes** | No | No | **YES** |
| 7 | `ssl_finetune_50pct` | SSL Only | 50% | 35 pts | **Yes** | No | No | **YES** |
| 8 | `ssl_finetune_100pct`| SSL Only | 100% | 70 pts | **Yes** | No | No | **YES** |
| 9 | `ssl_motion_10pct` | SSL + Motion (Ablation C)| 10% | 7 pts | **Yes** | **Yes** | No | **YES** |
| 10| `ssl_motion_25pct` | SSL + Motion | 25% | 17 pts | **Yes** | **Yes** | No | **YES** |
| 11| `ssl_motion_50pct` | SSL + Motion | 50% | 35 pts | **Yes** | **Yes** | No | **YES** |
| 12| `ssl_motion_100pct`| SSL + Motion | 100% | 70 pts | **Yes** | **Yes** | No | **YES** |
| 13| `ssl_pseudo_10pct` | SSL + Pseudo (Ablation D)| 10% | 7 pts | **Yes** | No | **Yes** | **YES** |
| 14| `full_pipeline_10pct`| Full Proposed (Ablation E)| 10% | 7 pts | **Yes** | **Yes** | **Yes** | **YES** |
| 15| `full_pipeline_25pct`| Full Proposed Framework | 25% | 17 pts | **Yes** | **Yes** | **Yes** | **YES** |
| 16| `full_pipeline_50pct`| Full Proposed Framework | 50% | 35 pts | **Yes** | **Yes** | **Yes** | **YES** |
| 17| `full_pipeline_100pct`| Full Proposed Framework | 100% | 70 pts | **Yes** | **Yes** | **Yes** | **YES** |
| 18| `robustness_seed123` | Full Pipeline (Seed 123) | 10% | 7 pts | **Yes** | **Yes** | **Yes** | **YES** |
| 19| `robustness_seed456` | Full Pipeline (Seed 456) | 10% | 7 pts | **Yes** | **Yes** | **Yes** | **YES** |

---

## 8. Expected Output Structure

Upon completion of training on the GPU cluster, outputs will populate the following directory layout:

```
results/
├── experiments/
│   └── experiment_registry.json      # Machine-readable JSON summary of all runs
├── tables/
│   ├── ablation_table.csv            # 5-variant component ablation results
│   ├── ablation_table.md             # Markdown table for publication/reports
│   ├── label_efficiency_table.csv    # 4-regime comparison table (10-100%)
│   ├── label_efficiency_table.md     # Markdown table
│   ├── patient_metrics_table.csv     # Individual test patient metrics (N=20)
│   ├── patient_metrics_table.md      # Markdown table
│   ├── robustness_table.csv          # Perturbations & multi-seed results
│   └── robustness_table.md           # Markdown table
└── figures/
    ├── label_efficiency_curves.png   # Dice vs. Annotation Fraction curves
    ├── ablation_comparison.png       # 5-variant component ablation bar plot
    ├── classwise_performance.png     # LV, MYO, RV performance comparison
    ├── temporal_consistency_plot.png # Frame-to-frame agreement & motion gain
    ├── pseudo_label_quality.png      # Confidence threshold calibration trade-off
    └── robustness_comparison.png     # Metric stability under perturbations
checkpoints/
├── baseline_unet_best.pth
├── ssl/
│   └── ssl_encoder_best.pth
├── motion/
│   └── motion_model_best.pth
└── experiments/
    └── {mode}_{fraction}pct_best.pth
```

---

## 9. Troubleshooting & Operational Notes

1. **CUDA Out of Memory (OOM)**:
   If running on a smaller GPU ($<6\text{ GB}$), reduce the batch size:
   In `configs/experiments.yaml`, change `batch_size: 8` to `batch_size: 4`.
2. **MedPy Fallback**:
   If `medpy` fails to install on your OS or architecture, `src/metrics.py` will automatically fall back to an internal SciPy Euclidean distance transform implementation (`_fallback_hd95`) to ensure continuous HD95 metric calculation without crashes.
3. **Reproducibility Guarantee**:
   All random seeds are fixed and deterministic algorithms are enabled in PyTorch (`torch.backends.cudnn.deterministic = True`). Patient splits are hardcoded in `data/splits/` to strictly prevent data leakage across environments.
