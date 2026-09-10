import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Motion-Guided Self-Supervised Learning for Cardiac Cine MRI\n",
                "## Notebook 06: Confidence-Filtered Pseudo-Labeling\n",
                "\n",
                "This notebook demonstrates the **Confidence-Filtered Pseudo-Labeling Pipeline** for limited-label semi-supervised cardiac segmentation.\n",
                "\n",
                "### Key Objectives:\n",
                "1. **Selective Acceptance**: Prevent model drift and error amplification by accepting only high-confidence predictions.\n",
                "2. **Multi-Metric Confidence**: Evaluate Maximum Softmax Probability (MSP), Shannon Entropy, and Motion-Warped Temporal Agreement.\n",
                "3. **Quality Assessment**: Measure acceptance rate, pixel accuracy on accepted regions, and per-class Dice against ground truth.\n",
                "4. **Persistent Export**: Save filtered pseudo-labels and metadata for downstream fine-tuning.\n",
                "\n",
                "> **COMPUTE CONSTRAINT NOTE**:\n",
                "> In accordance with project instructions, model training occurs on a separate GPU training system (`TRAINING MACHINE ONLY`).\n",
                "> This notebook runs a lightweight CPU demonstration illustrating confidence filtering and evaluation mechanics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "# Ensure project root is in sys.path\n",
                "project_root = Path.cwd().resolve()\n",
                "if project_root.name == 'notebooks':\n",
                "    project_root = project_root.parent\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.insert(0, str(project_root))\n",
                "\n",
                "import yaml\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import torch\n",
                "import torch.nn.functional as F\n",
                "\n",
                "from src.segmentation_model import SegmentationUNet\n",
                "from src.motion import MotionEstimator\n",
                "from src.dataset import ACDCSegDataset, ACDCTemporalDataset\n",
                "from src.pseudo_labels import (\n",
                "    compute_confidence_map,\n",
                "    compute_temporal_agreement,\n",
                "    apply_confidence_filtering,\n",
                "    evaluate_pseudo_label_quality,\n",
                ")\n",
                "\n",
                "print(f\"Project root: {project_root}\")\n",
                "print(f\"PyTorch version: {torch.__version__}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Load Pseudo-Label Configuration\n",
                "Inspect parameters defined in `configs/pseudo_labels.yaml`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "config_path = project_root / \"configs\" / \"pseudo_labels.yaml\"\n",
                "with open(config_path, \"r\") as f:\n",
                "    config = yaml.safe_load(f)\n",
                "\n",
                "print(\"Pseudo-Label Configuration:\")\n",
                "print(f\"  Confidence metric:    {config['filtering']['confidence_metric']}\")\n",
                "print(f\"  Confidence threshold: {config['filtering']['confidence_threshold']}\")\n",
                "print(f\"  Entropy threshold:    {config['filtering']['entropy_threshold']}\")\n",
                "print(f\"  Temporal consistency: {config['filtering']['use_temporal_consistency']}\")\n",
                "print(f\"  Ignore index:         {config['filtering']['ignore_index']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Load Cardiac MRI Slices and Ground Truth\n",
                "We load annotated slices from the validation split to illustrate predictions, confidence estimation, and evaluation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_cfg = config['data']\n",
                "val_dataset = ACDCSegDataset(\n",
                "    processed_dir=str(project_root / data_cfg['processed_dir']),\n",
                "    split_file=str(project_root / data_cfg['val_split']),\n",
                ")\n",
                "\n",
                "sample = val_dataset[10]\n",
                "image = sample['image'].unsqueeze(0) # (1, 1, 256, 256)\n",
                "gt_mask = sample['mask']             # (256, 256)\n",
                "\n",
                "print(f\"Patient:     {sample['patient_id']}\")\n",
                "print(f\"Slice Index: {sample['slice_idx']}\")\n",
                "print(f\"Phase:       {sample.get('phase', 'N/A')}\")\n",
                "print(f\"Image shape: {image.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Model Inference: Softmax Probabilities & Predictions\n",
                "We pass the cardiac slice through `SegmentationUNet` on CPU to obtain 4-class logits and softmax probabilities."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "device = torch.device('cpu')\n",
                "model = SegmentationUNet(\n",
                "    in_channels=1,\n",
                "    num_classes=4,\n",
                "    encoder_channels=[32, 64, 128, 256],\n",
                ").to(device)\n",
                "model.eval()\n",
                "\n",
                "with torch.no_grad():\n",
                "    logits = model(image)\n",
                "    probs = F.softmax(logits, dim=1)\n",
                "    raw_pseudo = torch.argmax(probs, dim=1)[0]\n",
                "\n",
                "print(f\"Logits shape:        {logits.shape}\")\n",
                "print(f\"Probabilities shape: {probs.shape}\")\n",
                "print(f\"Predicted classes:   {torch.unique(raw_pseudo).tolist()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Comparing Confidence Metrics\n",
                "We compare:\n",
                "1. **Maximum Softmax Probability (MSP)**: $C(x) = \\max_c P(y=c|x)$\n",
                "2. **Normalized Shannon Negative Entropy**: $C_{\\text{ent}}(x) = 1 + \\frac{\\sum_c P_c \\log P_c}{\\log C}$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "conf_msp = compute_confidence_map(probs, method=\"max_probability\")[0].numpy()\n",
                "conf_ent = compute_confidence_map(probs, method=\"entropy\")[0].numpy()\n",
                "\n",
                "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n",
                "axes[0].imshow(image.squeeze().numpy(), cmap='gray')\n",
                "axes[0].set_title(\"Input Cardiac MRI Slice\")\n",
                "axes[0].axis('off')\n",
                "\n",
                "im1 = axes[1].imshow(conf_msp, cmap='viridis', vmin=0.25, vmax=1.0)\n",
                "axes[1].set_title(f\"MSP Confidence Map\\n(Mean: {conf_msp.mean():.3f})\")\n",
                "axes[1].axis('off')\n",
                "plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)\n",
                "\n",
                "im2 = axes[2].imshow(conf_ent, cmap='plasma', vmin=0.0, vmax=1.0)\n",
                "axes[2].set_title(f\"Normalized Entropy Confidence\\n(Mean: {conf_ent.mean():.3f})\")\n",
                "axes[2].axis('off')\n",
                "plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Confidence Filtering: Accepted vs Rejected Pixels\n",
                "Applying a threshold (e.g. $\\tau = 0.35$ on initial weights) produces an acceptance mask where uncertain boundary/structure pixels are marked as `ignore_index` (-1)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tau = 0.35  # Threshold for illustration\n",
                "filtered_labels, accept_mask, acc_rate = apply_confidence_filtering(\n",
                "    pseudo_labels=raw_pseudo.unsqueeze(0),\n",
                "    confidence=torch.from_numpy(conf_msp).unsqueeze(0),\n",
                "    threshold=tau,\n",
                "    ignore_index=-1,\n",
                ")\n",
                "\n",
                "filt_np = filtered_labels[0].numpy()\n",
                "mask_np = accept_mask[0].numpy()\n",
                "\n",
                "fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))\n",
                "\n",
                "axes[0].imshow(image.squeeze().numpy(), cmap='gray')\n",
                "axes[0].set_title(\"Input Image\")\n",
                "axes[0].axis('off')\n",
                "\n",
                "axes[1].imshow(raw_pseudo.numpy(), cmap='tab10', vmin=0, vmax=3)\n",
                "axes[1].set_title(\"Raw Unfiltered Predictions\")\n",
                "axes[1].axis('off')\n",
                "\n",
                "axes[2].imshow(mask_np, cmap='Blues', alpha=0.9)\n",
                "axes[2].set_title(f\"Acceptance Mask (tau={tau})\\nAccepted: {acc_rate*100:.1f}%\")\n",
                "axes[2].axis('off')\n",
                "\n",
                "# Display filtered labels (masking out rejected pixels in black)\n",
                "filt_display = np.where(filt_np == -1, np.nan, filt_np)\n",
                "axes[3].imshow(image.squeeze().numpy(), cmap='gray')\n",
                "axes[3].imshow(filt_display, cmap='tab10', vmin=0, vmax=3, alpha=0.7)\n",
                "axes[3].set_title(\"Filtered Pseudo-Labels\\n(Uncertain Pixels Excluded)\")\n",
                "axes[3].axis('off')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Temporal Consistency Agreement with Motion Warping\n",
                "When temporal pairs $(I_t, I_{t+1})$ are available, we use `SimpleFlowNet` from Prompt 6 to forward-warp predictions and verify agreement."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "temp_dataset = ACDCTemporalDataset(\n",
                "    processed_dir=str(project_root / data_cfg['processed_dir']),\n",
                "    split_file=str(project_root / data_cfg['val_split']),\n",
                ")\n",
                "t_sample = temp_dataset[0]\n",
                "frame_t = t_sample['frame_t'].unsqueeze(0)\n",
                "frame_t1 = t_sample['frame_t1'].unsqueeze(0)\n",
                "\n",
                "motion_est = MotionEstimator(channels=[16, 32, 64, 32]).to(device).eval()\n",
                "with torch.no_grad():\n",
                "    p_t = F.softmax(model(frame_t), dim=1)\n",
                "    p_t1 = F.softmax(model(frame_t1), dim=1)\n",
                "    flow = motion_est(frame_t, frame_t1)['flow']\n",
                "    agreement = compute_temporal_agreement(p_t, p_t1, flow, motion_est.transformer)\n",
                "\n",
                "agree_np = agreement[0].numpy()\n",
                "print(f\"Mean Temporal Agreement: {agree_np.mean():.4f}\")\n",
                "\n",
                "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\n",
                "axes[0].imshow(frame_t.squeeze().numpy(), cmap='gray')\n",
                "axes[0].set_title(\"Frame t\")\n",
                "axes[0].axis('off')\n",
                "\n",
                "axes[1].imshow(frame_t1.squeeze().numpy(), cmap='gray')\n",
                "axes[1].set_title(\"Frame t+1\")\n",
                "axes[1].axis('off')\n",
                "\n",
                "im = axes[2].imshow(agree_np, cmap='coolwarm', vmin=0.5, vmax=1.0)\n",
                "axes[2].set_title(f\"Motion Agreement Map\\n(Mean: {agree_np.mean():.3f})\")\n",
                "axes[2].axis('off')\n",
                "plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Pseudo-Label Quality Evaluation Against Ground Truth\n",
                "We evaluate pseudo-label metrics across a sweep of confidence thresholds to inspect the acceptance rate vs accuracy trade-off."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]\n",
                "acc_rates = []\n",
                "accuracies = []\n",
                "\n",
                "for t in thresholds:\n",
                "    _, m, rate = apply_confidence_filtering(\n",
                "        pseudo_labels=raw_pseudo.unsqueeze(0),\n",
                "        confidence=torch.from_numpy(conf_msp).unsqueeze(0),\n",
                "        threshold=t,\n",
                "    )\n",
                "    metrics = evaluate_pseudo_label_quality(\n",
                "        pseudo_labels=raw_pseudo,\n",
                "        ground_truth=gt_mask,\n",
                "        accept_mask=m[0],\n",
                "    )\n",
                "    acc_rates.append(rate * 100)\n",
                "    accuracies.append(metrics['accuracy_on_accepted'] * 100)\n",
                "\n",
                "fig, ax1 = plt.subplots(figsize=(8, 4.5))\n",
                "ax1.plot(thresholds, acc_rates, 'b-o', label='Acceptance Rate (%)')\n",
                "ax1.set_xlabel('Confidence Threshold (tau)')\n",
                "ax1.set_ylabel('Acceptance Rate (%)', color='b')\n",
                "ax1.tick_params(axis='y', labelcolor='b')\n",
                "ax1.grid(True, linestyle='--', alpha=0.5)\n",
                "\n",
                "ax2 = ax1.twinx()\n",
                "ax2.plot(thresholds, accuracies, 'r-s', label='Accepted Accuracy (%)')\n",
                "ax2.set_ylabel('Accuracy on Accepted Pixels (%)', color='r')\n",
                "ax2.tick_params(axis='y', labelcolor='r')\n",
                "\n",
                "plt.title('Threshold vs Acceptance Rate & Quality Trade-Off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8. Running on GPU Training System (`TRAINING MACHINE ONLY`)\n",
                "\n",
                "To generate full-cohort pseudo-labels on the separate GPU training server:\n",
                "\n",
                "```bash\n",
                "# ========================================================\n",
                "# TRAINING MACHINE ONLY (DO NOT RUN ON DEV SYSTEM)\n",
                "# ========================================================\n",
                "python src/pseudo_labels.py --config configs/pseudo_labels.yaml --device cuda\n",
                "```\n",
                "\n",
                "The filtered pseudo-labels and metadata will be saved to `results/pseudo_labels/`."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebooks/06_confidence_pseudo_labels.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated {out_path}")
