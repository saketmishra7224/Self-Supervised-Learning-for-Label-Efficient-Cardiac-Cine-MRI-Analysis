"""nnU-Net v2 baseline integration (external self-configuring pipeline).

"nnU-Net baseline: same ACDC patient split and same held-out test set, with nnU-Net's standard configuration; predictions are passed through our common evaluation pipeline."

This module is a thin bridge between the project's preprocessed ACDC data
(.npz slices, project label mapping BG=0/LV=1/MYO=2/RV=3) and the official
nnU-Net v2 workflow (planning/preprocessing/training/prediction). nnU-Net
itself is never reimplemented here: training and inference run through the
official ``nnUNetv2_*`` commands with nnU-Net-native configuration (its own
optimizer, schedule, augmentation, and 2D planning). The project's AdamW
setup is NOT forced onto nnU-Net.

Unavoidable adaptations from the standard nnU-Net workflow (documented):
1. Each training case is a single labeled 2D slice exported as a
   (256, 256, 1) NIfTI file, because the benchmark trains/evaluates 2D
   slices (only the 2D configuration is planned/run via ``-c 2d``).
2. Through-plane spacing metadata is set to 10.0 mm (slice thickness is not
   stored in the preprocessed archive). This informs nnU-Net planning only
   and never alters voxel values or in-plane geometry.
3. ``splits_final.json`` pins the project's exact 70-train / 10-val patient
   split instead of nnU-Net's default random 5-fold splits. The 20 test
   patients appear only under ``imagesTs/`` (images only, no labels).
4. Training-budget deviation (explicit, no budget chosen here): nnU-Net's
   standard configuration (1000 epochs x 5-fold cross-validation) exceeds
   Kaggle GPU quota and therefore cannot be run as-is. Any executed
   epoch/fold budget is a documented deviation that must be fixed in the
   training plan and recorded in run metadata before training; this module
   does not invent or assume one.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

DATASET_ID = 701
DATASET_NAME = "ACDCHeart"
EXPECTED_LABELS = {0, 1, 2, 3}
IN_PLANE_SPACING = (1.5, 1.5)
THROUGH_PLANE_SPACING = 10.0
CASE_PATTERN = re.compile(r"(patient\d+)_frame(\d+)_slice(\d+)")


def read_patient_list(split_file: str) -> List[str]:
    """Read patient IDs deterministically (sorted, de-duplicated)."""
    with open(split_file) as f:
        ids = sorted({line.strip() for line in f if line.strip()})
    if not ids:
        raise ValueError(f"No patient IDs in {split_file}")
    return ids


def parse_case_stem(stem: str) -> Tuple[str, int, int]:
    """Parse 'patient001_frame00_slice05' -> (patient001, 0, 5)."""
    match = CASE_PATTERN.fullmatch(stem)
    if not match:
        raise ValueError(f"Filename does not match case pattern: {stem}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def load_labeled_slice(npz_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load one slice; return (image f32, mask int64, meta). Verifies labels."""
    with np.load(npz_path, allow_pickle=True) as data:
        image = np.asarray(data["image"], dtype=np.float32)
        mask = np.asarray(data["mask"], dtype=np.int64)
        meta = {
            "patient_id": str(data["patient_id"]),
            "slice_idx": int(data["slice_idx"]),
            "frame_idx": int(data["frame_idx"]),
        }
    if image.shape != (256, 256) or mask.shape != (256, 256):
        raise ValueError(f"Unexpected shapes in {npz_path}")
    values = set(np.unique(mask).tolist())
    if (mask == -1).all():
        raise ValueError(f"Slice has no labels (all -1): {npz_path}")
    unexpected = values - EXPECTED_LABELS
    if unexpected:
        raise ValueError(f"Unexpected label values {sorted(unexpected)} in {npz_path}")
    return image, mask, meta


def write_case_nifti(array2d: np.ndarray, path: str, is_label: bool) -> None:
    """Write a 2D slice as a (H, W, 1) NIfTI case nnU-Net can read as 3D."""
    if is_label:
        arr = np.ascontiguousarray(array2d.astype(np.uint8))[:, :, None]
    else:
        arr = np.ascontiguousarray(array2d.astype(np.float32))[:, :, None]
    affine = np.diag([IN_PLANE_SPACING[0], IN_PLANE_SPACING[1],
                      THROUGH_PLANE_SPACING, 1.0])
    nib.save(nib.Nifti1Image(arr, affine), path)


def export_dataset(
    processed_dir: str,
    splits_dir: str,
    raw_root: str,
    dataset_id: int = DATASET_ID,
    dataset_name: str = DATASET_NAME,
    limit_patients_per_split: Optional[Dict[str, int]] = None,
) -> Dict:
    """Export labeled slices to nnU-Net raw format. Returns a summary dict.

    Writes ``Dataset{id}_{name}/`` with ``imagesTr/``, ``labelsTr/``,
    ``imagesTs/``, ``dataset.json`` and ``case_mapping.json``. Test patients
    contribute images only (no labels exported). Set
    ``limit_patients_per_split`` (e.g. ``{"train": 2}``) for a dry-run
    subset; ``None`` exports the full benchmark cohort.
    """
    processed, splits = Path(processed_dir), Path(splits_dir)
    train_pts = read_patient_list(str(splits / "train_patients.txt"))
    val_pts = read_patient_list(str(splits / "val_patients.txt"))
    test_pts = read_patient_list(str(splits / "test_patients.txt"))
    if set(train_pts) & set(val_pts) or set(train_pts) & set(test_pts) or set(val_pts) & set(test_pts):
        raise ValueError("Project splits overlap; refusing export.")
    if limit_patients_per_split:
        train_pts = train_pts[: limit_patients_per_split.get("train", len(train_pts))]
        val_pts = val_pts[: limit_patients_per_split.get("val", len(val_pts))]
        test_pts = test_pts[: limit_patients_per_split.get("test", len(test_pts))]

    dest = Path(raw_root) / f"Dataset{dataset_id}_{dataset_name}"
    images_tr, labels_tr, images_ts = (dest / d for d in ("imagesTr", "labelsTr", "imagesTs"))
    for d in (images_tr, labels_tr, images_ts):
        d.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, dict] = {}
    counts = {"train": 0, "val": 0, "test": 0}
    pool = ([(p, "train") for p in train_pts] + [(p, "val") for p in val_pts]
            + [(p, "test") for p in test_pts])
    for patient, split in pool:
        for npz_path in sorted(Path(processed_dir).glob(f"{patient}_frame*_slice*.npz")):
            pid, frame, sl = parse_case_stem(npz_path.stem)
            assert pid == patient, f"ID mismatch: {npz_path}"
            with np.load(npz_path, allow_pickle=True) as data:
                image = np.asarray(data["image"], dtype=np.float32)
                mask = np.asarray(data["mask"], dtype=np.int64)
            if image.shape != (256, 256) or mask.shape != (256, 256):
                raise ValueError(f"Unexpected shapes in {npz_path}")
            if (mask == -1).all():
                continue  # unlabeled cine frame: not part of the benchmark
            unexpected = set(np.unique(mask).tolist()) - EXPECTED_LABELS
            if unexpected:
                raise ValueError(f"Unexpected labels {sorted(unexpected)} in {npz_path}")
            case_id = f"{pid}_f{frame:02d}_s{sl:02d}"
            write_case_nifti(image, str(images_tr / f"{case_id}_0000.nii.gz")
                             if split != "test" else str(images_ts / f"{case_id}_0000.nii.gz"),
                             is_label=False)
            if split != "test":
                write_case_nifti(mask, str(labels_tr / f"{case_id}.nii.gz"), is_label=True)
            mapping[case_id] = {"patient_id": pid, "frame_idx": frame,
                                "slice_idx": sl, "split": split}
            counts[split] += 1

    dataset_json = {
        "channel_names": {"0": "cineMRI"},
        "labels": {"background": 0, "LV": 1, "myocardium": 2, "RV": 3},
        "numTraining": counts["train"] + counts["val"],
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
    }
    (dest / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    (dest / "case_mapping.json").write_text(json.dumps(mapping, indent=2))
    return {"dataset_dir": str(dest), "counts": counts,
            "n_train_patients": len(train_pts), "n_val_patients": len(val_pts),
            "n_test_patients": len(test_pts)}


def write_splits_final(mapping: Dict[str, dict], preprocessed_dataset_dir: str) -> str:
    """Write nnU-Net ``splits_final.json`` pinning the project train/val split.

    ``mapping`` is the case mapping produced by :func:`export_dataset`.
    The file is written into nnU-Net's preprocessed dataset directory (where
    the trainer looks for it); only the filename/location follows nnU-Net
    convention, the patient assignment follows the project splits exactly.
    """
    train_ids = sorted(cid for cid, info in mapping.items() if info["split"] == "train")
    val_ids = sorted(cid for cid, info in mapping.items() if info["split"] == "val")
    if not train_ids or not val_ids:
        raise ValueError("Need non-empty train and val case lists for splits_final.json.")
    out = Path(preprocessed_dataset_dir) / "splits_final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{"train": train_ids, "val": val_ids}], indent=2))
    return str(out)


def adapt_predictions(
    pred_dir: str,
    mapping: Dict[str, dict],
    processed_dir: str,
) -> Dict:
    """Convert nnU-Net test predictions to the common evaluator format.

    Reads one ``<case_id>.nii.gz`` prediction per test case (as written by
    ``nnUNetv2_predict``), aligns each to its ground-truth slice, and runs
    the project's ``compute_metrics_batch`` + ``compute_patient_level_metrics``
    (Dice + HD95, voxel spacing 1.5 mm). Returns the patient-level results.
    """
    from src.metrics import compute_metrics_batch, compute_patient_level_metrics

    preds, targets, patient_ids = [], [], []
    for case_id in sorted(mapping):
        info = mapping[case_id]
        if info["split"] != "test":
            continue
        pred = np.ascontiguousarray(
            np.asarray(nib.load(str(Path(pred_dir) / f"{case_id}.nii.gz")).get_fdata()).squeeze())
        if pred.shape != (256, 256):
            raise ValueError(f"Prediction {case_id} has shape {pred.shape}, expected (256, 256)")
        pred = pred.astype(np.int64)
        unexpected = set(np.unique(pred).tolist()) - EXPECTED_LABELS
        if unexpected:
            raise ValueError(f"Prediction {case_id} has labels {sorted(unexpected)}")
        npz = (f"{info['patient_id']}_frame{info['frame_idx']:02d}"
               f"_slice{info['slice_idx']:02d}.npz")
        with np.load(str(Path(processed_dir) / npz), allow_pickle=True) as data:
            target = np.asarray(data["mask"], dtype=np.int64)
        preds.append(pred)
        targets.append(target)
        patient_ids.append(info["patient_id"])
    if not preds:
        raise ValueError(f"No test predictions found in {pred_dir}")
    batch = compute_metrics_batch(np.stack(preds), np.stack(targets), patient_ids,
                                  voxel_spacing=(1.5, 1.5), compute_hd=True)
    return compute_patient_level_metrics(batch["per_sample"])


def main() -> None:
    parser = argparse.ArgumentParser(description="nnU-Net v2 baseline integration")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="Export project slices to nnU-Net raw format")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--raw-root", required=True)
    p.add_argument("--dataset-id", type=int, default=DATASET_ID)
    p.add_argument("--dataset-name", default=DATASET_NAME)
    args = parser.parse_args()
    if args.command == "export":
        summary = export_dataset(args.processed_dir, args.splits_dir,
                                 args.raw_root, args.dataset_id, args.dataset_name)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
