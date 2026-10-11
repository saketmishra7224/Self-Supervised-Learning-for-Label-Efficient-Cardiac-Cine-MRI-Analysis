"""MedSAM zero-shot inference + scoring runner (no training).

Runs the frozen MedSAM baseline (bowang-lab/MedSAM, ViT-B) over the locked
20-patient TEST split with the versioned training-only box priors
(``configs/medsam_prior_boxes.json``) and scores predictions with the
project's common metric utilities (Dice + HD95, patient-level aggregation).

Safety policy (locked-test safeguards preserved):
  - MedSAM is NEVER trained here (``torch.no_grad()``, frozen weights).
  - Prompts are the three fixed population boxes loaded from the
    version-controlled training-only artifact; they do not depend on the
    input image, so test ground truth cannot leak by construction. The exact
    boxes used are recorded in the output JSON for provenance.
  - No thresholds or prompts are tuned (fixed argmax over pseudo-logits).
  - Full runs refuse to write unless the evaluated cohort exactly matches
    the canonical test cohort (same gate as ``src/evaluate_test.py``).
  - ``--max-slices`` runs only the first N test slices deterministically
    for quick calibration; its output is explicitly labeled partial and
    skips the cohort gate (a subset can never match the full cohort).

MedSAM has no trainable checkpoint, so it is intentionally NOT scored
through ``src/evaluate_test.py`` (which requires ``experiment_name`` +
``model_state_dict`` checkpoint provenance).

Usage:
    python -m src.run_medsam_baseline --device cuda
    python -m src.run_medsam_baseline --max-slices 10 --output-dir /tmp/medsam_pilot
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import ACDCSegDataset  # noqa: E402
from src.medsam_baseline import (  # noqa: E402
    CLASSES,
    build_medsam,
    default_prior_boxes_path,
    load_prior_boxes,
)
from src.metrics import compute_metrics_batch, compute_patient_level_metrics  # noqa: E402
from src.evaluate_test import CANONICAL_TEST_SPLIT  # noqa: E402

RUN_ID = "medsam_zero_shot"
VOXEL_SPACING = (1.5, 1.5)
EXPECTED_LABELS = {0, 1, 2, 3}


def parse_args():
    parser = argparse.ArgumentParser(description="MedSAM zero-shot test inference + scoring (no training)")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--test-split", default="data/splits/test_patients.txt")
    parser.add_argument("--output-dir", default="results/baseline/medsam_zero_shot")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Official medsam_vit_b.pth (default: ~/.cache/medsam/MedSAM/ or $MEDSAM_CHECKPOINT)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="DataLoader batch size (forward runs per slice internally)")
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Calibration only: score the first N test slices; output labeled partial")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    # Train-only priors from the version-controlled artifact (never invented,
    # never derived from test labels). Recorded verbatim in the output.
    prior_path = default_prior_boxes_path()
    prior_boxes = load_prior_boxes()
    print(f"Prior boxes: {prior_path} (train-only, {len(prior_boxes)} classes)")

    model = build_medsam(checkpoint_path=args.checkpoint_path, device=device)
    print(f"MedSAM weights: {model.checkpoint_used} (frozen zero-shot)")
    assert model.prompts is not None, "MedSAM runner requires attached box prompts."

    dataset = ACDCSegDataset(
        processed_dir=args.processed_dir,
        split_file=args.test_split,
        transform=None,  # raw slices; MedSAM scales internally per slice
    )
    if len(dataset) == 0:
        raise ValueError(f"Test dataset is empty: {args.test_split}")
    partial = args.max_slices is not None
    if partial:
        dataset = Subset(dataset, list(range(min(args.max_slices, len(dataset)))))
        print(f"CALIBRATION MODE: first {len(dataset)} slices only; output will be labeled partial.")
    print(f"Test slices: {len(dataset)} (patients: {dataset.dataset.inner.patient_ids if isinstance(dataset, Subset) else dataset.get_patient_ids()})")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    all_preds, all_targets, all_pids = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            assert tuple(images.shape[2:]) == (256, 256), tuple(images.shape)
            logits = model(images).float()  # (B, 4, H, W) pseudo-logits, CPU
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            unexpected = set(np.unique(preds).tolist()) - EXPECTED_LABELS
            if unexpected:
                raise ValueError(f"MedSAM predictions contain labels {sorted(unexpected)}")
            all_preds.append(preds)
            all_targets.append(np.asarray(batch["mask"]))
            all_pids.extend([str(p) for p in batch["patient_id"]])
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    batch_results = compute_metrics_batch(
        all_preds, all_targets, all_pids,
        voxel_spacing=VOXEL_SPACING, compute_hd=True,
    )
    patient_results = compute_patient_level_metrics(batch_results["per_sample"])

    if not partial:
        expected = {ln.strip() for ln in Path(CANONICAL_TEST_SPLIT).read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")}
        evaluated = set(patient_results["per_patient"])
        if evaluated != expected:
            raise ValueError(
                f"Evaluated cohort does not match canonical test cohort: missing="
                f"{sorted(expected - evaluated)}, unexpected={sorted(evaluated - expected)}. "
                "Refusing to write metrics."
            )

    output = {
        "run_id": RUN_ID,
        "trainable": False,
        "partial": partial,
        "checkpoint_used": model.checkpoint_used,
        "prior_boxes_path": str(prior_path),
        "prior_boxes_256": {str(k): [float(v) for v in vals] for k, vals in prior_boxes.items()},
        "prompt_classes": {str(c): name for c, name in zip(CLASSES, ["LV", "myocardium", "RV"])},
        "test_split": str(args.test_split),
        "n_patients": patient_results["n_patients"],
        "n_slices": int(all_preds.shape[0]),
        "voxel_spacing": list(VOXEL_SPACING),
        "per_patient": patient_results["per_patient"],
        "mean": patient_results["mean"],
        "std": patient_results["std"],
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "medsam_zeroshot_metrics.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    mean = patient_results["mean"]
    print(f"MedSAM zero-shot ({'PARTIAL' if partial else 'FULL'}): "
          f"{patient_results['n_patients']} patients, {all_preds.shape[0]} slices, "
          f"Mean_Dice={mean['Mean_Dice']:.4f}")


if __name__ == "__main__":
    main()
