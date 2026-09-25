"""Final single-pass patient-level TEST evaluation for Full Pipeline runs.

Reads ONLY the fixed 20-patient ACDC test split. Performs no checkpoint
selection or threshold tuning: each run is scored once from its *_best.pth
checkpoint, then results are written to that run's results directory.

Usage:
    python -m src.evaluate_test --run-id full_pipeline_10pct_seed42
    python -m src.evaluate_test --all
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ACDCSegDataset, get_val_transforms
from src.metrics import compute_metrics_batch, compute_patient_level_metrics
from src.segmentation_model import SegmentationUNet

# Physical in-plane spacing (mm) of the preprocessed 256x256 volumes.
# Required so HD95 is reported in mm as documented; the training-time
# validation path passes voxel_spacing=None (pixel units).
VOXEL_SPACING = (1.5, 1.5)

# The only production runs this evaluator may score. Smoke runs
# (e.g. full_pipeline_10pct_seed42_smoke) are never matched.
PRODUCTION_RUN_IDS = [
    "full_pipeline_10pct_seed42",
    "full_pipeline_25pct_seed42",
    "full_pipeline_50pct_seed42",
    "full_pipeline_100pct_seed42",
]

_RUN_ID_PATTERN = re.compile(r"^full_pipeline_(\d+)pct_seed(\d+)$")

# Canonical fixed 20-patient test cohort. Results may only be written when
# the evaluated patient IDs match this cohort exactly.
CANONICAL_TEST_SPLIT = Path("data/splits/test_patients.txt")


def _read_cohort_ids(split_file: Path) -> set:
    return {
        line.strip()
        for line in Path(split_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def parse_run_id(run_id: str) -> tuple:
    """Return (label_fraction, seed) for a production run ID or raise."""
    match = _RUN_ID_PATTERN.match(run_id)
    if match is None or run_id not in PRODUCTION_RUN_IDS:
        raise ValueError(
            f"Refusing to evaluate unknown run '{run_id}'. "
            f"Allowed run IDs: {PRODUCTION_RUN_IDS}"
        )
    return int(match.group(1)), int(match.group(2))


def build_model_from_config(config: dict, device: torch.device) -> SegmentationUNet:
    """Construct the training-time SegmentationUNet from experiments.yaml."""
    model_cfg = config.get("model", {})
    return SegmentationUNet(
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 4),
        encoder_channels=model_cfg.get("encoder_channels", [32, 64, 128, 256]),
        dropout=model_cfg.get("dropout", 0.1),
        use_residual=model_cfg.get("use_residual", True),
    ).to(device)


def evaluate_run(
    run_id: str,
    config: dict,
    device: torch.device,
    batch_size: int = 8,
    processed_dir_override=None,
) -> dict:
    """Score one production run on the fixed test split (single pass)."""
    label_fraction, seed = parse_run_id(run_id)
    data_cfg = config.get("data", {})
    log_cfg = config.get("logging", {})

    checkpoint_path = (
        Path(log_cfg.get("checkpoint_dir", "checkpoints/experiments")) / run_id / f"{run_id}_best.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Production checkpoint not found: {checkpoint_path}. "
            "Only *_best.pth checkpoints are evaluated."
        )
    test_split = data_cfg.get("test_split", "data/splits/test_patients.txt")

    model = build_model_from_config(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    provenance = checkpoint.get("experiment_name")
    if provenance is None or str(provenance) != run_id:
        raise ValueError(
            f"Refusing to evaluate {checkpoint_path}: checkpoint provenance "
            f"'{provenance}' does not match requested run '{run_id}'. "
            "Only the run's own *_best.pth checkpoint may be evaluated."
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    processed_dir = processed_dir_override or data_cfg.get("processed_dir", "data/processed")

    dataset = ACDCSegDataset(
        processed_dir,
        test_split,
        transform=get_val_transforms(),
    )
    if len(dataset) == 0:
        raise ValueError(
            f"Test dataset is empty: {test_split} | processed_dir={processed_dir}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    all_preds, all_targets, all_patient_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {run_id} [Test]", leave=False):
            images = batch["image"].to(device)
            logits = model(images)
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.append(batch["mask"].cpu().numpy())
            all_patient_ids.extend(list(batch["patient_id"]))
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    batch_results = compute_metrics_batch(
        all_preds,
        all_targets,
        all_patient_ids,
        voxel_spacing=VOXEL_SPACING,
        compute_hd=True,
    )
    patient_results = compute_patient_level_metrics(batch_results["per_sample"])

    expected_cohort = _read_cohort_ids(CANONICAL_TEST_SPLIT)
    evaluated_cohort = set(patient_results["per_patient"])
    if evaluated_cohort != expected_cohort:
        raise ValueError(
            "Evaluated patient cohort does not match the canonical fixed "
            f"test cohort ({CANONICAL_TEST_SPLIT}): missing="
            f"{sorted(expected_cohort - evaluated_cohort)}, unexpected="
            f"{sorted(evaluated_cohort - expected_cohort)}. "
            "Refusing to write test_metrics.json."
        )

    output = {
        "run_id": run_id,
        "label_fraction": label_fraction,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "test_split": str(test_split),
        "n_patients": patient_results["n_patients"],
        "n_slices": len(dataset),
        "voxel_spacing": list(VOXEL_SPACING),
        "per_patient": patient_results["per_patient"],
        "mean": patient_results["mean"],
        "std": patient_results["std"],
    }

    results_dir = Path(log_cfg.get("results_dir", "results/experiments")) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "test_metrics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return output


def print_summary(result: dict) -> None:
    """Print mean +- std for the required metric set."""
    mean, std = result["mean"], result["std"]
    print(f"\n{result['run_id']}: {result['n_patients']} patients, "
          f"{result['n_slices']} slices")
    for key in ["LV_Dice", "Myocardium_Dice", "RV_Dice", "Mean_Dice",
                "LV_HD95", "Myocardium_HD95", "RV_HD95", "Mean_HD95"]:
        print(f"  {key:18s}: {mean[key]:.4f} +- {std[key]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final patient-level TEST evaluation for Full Pipeline runs"
    )
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-id", default=None,
                        help="Production run ID to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all four production Full Pipeline runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Override the processed ACDC data directory used for final test evaluation.",
    )
    args = parser.parse_args()

    if args.all:
        run_ids = list(PRODUCTION_RUN_IDS)
    elif args.run_id is not None:
        run_ids = [args.run_id]
    else:
        parser.error("Provide --run-id <run_id> or --all.")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    for run_id in run_ids:
        print_summary(evaluate_run(run_id, config, device, args.batch_size, args.processed_dir))


if __name__ == "__main__":
    main()
