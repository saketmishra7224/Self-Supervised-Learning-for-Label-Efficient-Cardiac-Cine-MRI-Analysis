"""
ACDC Dataset Inspector and Verification Utilities.

Provides reusable utilities for:
1. Verifying directory structure and dataset presence
2. Inspecting patient metadata (Info.cfg), 4D cine, ED/ES frames, and ground truth
3. Distinguishing labeled vs unlabeled temporal frames
4. Calculating dataset-wide statistics (dimensions, spacing, class distribution)
5. Validating file integrity and labels
6. Generating machine-readable inventory (data/processed/dataset_inventory.csv)
7. Generating publication-quality exploratory figures (results/data_exploration/)

Follows project specification:
Labels: 0=Background, 1=LV cavity, 2=Myocardium, 3=RV cavity
"""

import os
import re
import csv
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Expected ACDC original labels: 0=BG, 1=RV, 2=Myocardium, 3=LV
# Project specification labels:  0=BG, 1=LV, 2=Myocardium, 3=RV
ACDC_TO_SPEC_LABEL_MAP = {0: 0, 1: 3, 2: 2, 3: 1}
SPEC_LABEL_NAMES = {
    0: "Background",
    1: "LV cavity",
    2: "Myocardium",
    3: "RV cavity"
}
ACDC_ORIGINAL_LABEL_NAMES = {
    0: "Background",
    1: "RV cavity",
    2: "Myocardium",
    3: "LV cavity"
}
PATHOLOGY_MAP = {
    "NOR": "Normal",
    "MINF": "Previous Myocardial Infarction",
    "DCM": "Dilated Cardiomyopathy",
    "HCM": "Hypertrophic Cardiomyopathy",
    "ARV": "Abnormal Right Ventricle"
}


def get_expected_acdc_structure() -> str:
    """Return the exact expected directory structure for ACDC dataset."""
    return """
Expected Directory Structure:
=============================
data/raw/ACDC/
  |-- training/
  |     |-- patient001/
  |     |     |-- Info.cfg                     # Metadata: Group, Height, Weight, ED, ES
  |     |     |-- patient001_4d.nii.gz         # Complete 4D Cine volume (X, Y, Z, T)
  |     |     |-- patient001_frame01.nii.gz    # 3D volume at ED frame
  |     |     |-- patient001_frame01_gt.nii.gz # Ground-truth mask at ED frame
  |     |     |-- patient001_frame12.nii.gz    # 3D volume at ES frame
  |     |     `-- patient001_frame12_gt.nii.gz # Ground-truth mask at ES frame
  |     |-- patient002/
  |     |     `-- ...
  |     `-- patient100/
  |           `-- ... (100 patients: 20 per pathology group)
  `-- testing/ (optional / challenge evaluation)
        |-- patient101/
        `-- ... patient150/

Official Source Information:
----------------------------
- Challenge: Automated Cardiac Diagnosis Challenge (ACDC), hosted at MICCAI 2017.
- Official Portal: Human Heart Project / CREATIS platform (ACDC Challenge).
- Dataset package to download: 'training.zip' (and optionally 'testing.zip').
- Training set contains 100 patients (patient001 to patient100) evenly distributed
  across 5 groups: NOR (Normal), MINF (Myocardial Infarction), DCM (Dilated Cardiomyopathy),
  HCM (Hypertrophic Cardiomyopathy), ARV (Abnormal Right Ventricle).
- Unzip the archive directly into: data/raw/ACDC/
  such that data/raw/ACDC/training/patient001/Info.cfg exists.
"""


def parse_info_cfg(info_path: str) -> Dict[str, Any]:
    """
    Parse an ACDC Info.cfg file into a dictionary.
    Keys typically include: Group, Height, Weight, ED, ES.
    """
    info: Dict[str, Any] = {}
    if not os.path.exists(info_path):
        return info
    
    with open(info_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                info[key] = val
    return info


def remap_labels_to_spec(mask: Any) -> Any:
    """
    Remap original ACDC segmentation labels to project specification:
    Original: 0=BG, 1=RV, 2=Myo, 3=LV
    Target:   0=BG, 1=LV, 2=Myo, 3=RV
    """
    import numpy as np
    remapped = np.zeros_like(mask)
    for orig_val, spec_val in ACDC_TO_SPEC_LABEL_MAP.items():
        remapped[mask == orig_val] = spec_val
    return remapped


class ACDCInspector:
    """
    Comprehensive inspector for ACDC dataset.
    Can inspect dataset files, generate inventories, run quality checks,
    and output summary reports.
    """
    
    def __init__(self, raw_dir: str = "data/raw/ACDC"):
        self.raw_dir = Path(raw_dir)
        self.training_dir = self.raw_dir / "training"
        self.testing_dir = self.raw_dir / "testing"
    
    def verify_presence(self) -> Tuple[bool, str]:
        """
        Verify if the ACDC dataset is present at the configured location.
        Returns (is_present, message).
        """
        if not self.raw_dir.exists():
            return False, f"Directory does not exist: {self.raw_dir}"
        
        training_exists = self.training_dir.exists()
        direct_patients = [d for d in self.raw_dir.glob("patient*") if d.is_dir()]
        training_patients = [d for d in self.training_dir.glob("patient*") if d.is_dir()] if training_exists else []
        
        if not direct_patients and not training_patients:
            return False, (
                f"No patient directories (patientXXX) found in '{self.raw_dir}' or '{self.training_dir}'.\n"
                f"{get_expected_acdc_structure()}"
            )
        
        count = len(training_patients) if training_patients else len(direct_patients)
        effective_dir = self.training_dir if training_patients else self.raw_dir
        return True, f"Found {count} patient directories in {effective_dir}"

    def get_patient_dirs(self, subset: str = "training") -> List[Path]:
        """Get sorted list of patient directories."""
        target_dir = self.training_dir if subset == "training" else self.testing_dir
        if not target_dir.exists():
            patients = sorted([d for d in self.raw_dir.iterdir() if d.is_dir() and d.name.startswith("patient")])
            return patients
        return sorted([d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("patient")])

    def inspect_patient(self, pdir: Path, load_data: bool = True) -> Dict[str, Any]:
        """
        Thoroughly inspect a single patient folder.
        """
        pid = pdir.name
        info_file = pdir / "Info.cfg"
        info = parse_info_cfg(str(info_file)) if info_file.exists() else {}
        
        group = info.get("Group", "Unknown")
        ed_frame = info.get("ED", None)
        es_frame = info.get("ES", None)
        height = info.get("Height", None)
        weight = info.get("Weight", None)
        
        cine_4d_files = list(pdir.glob(f"{pid}_4d.nii.gz"))
        if not cine_4d_files:
            cine_4d_files = list(pdir.glob("*_4d.nii.gz"))
        has_4d = len(cine_4d_files) > 0
        
        num_temporal_frames = 0
        volume_shape = (None, None, None)
        spacing = (None, None, None)
        is_corrupted = False
        corruption_error = ""
        
        if has_4d and load_data:
            try:
                import nibabel as nib
                import numpy as np

                img_4d = nib.load(str(cine_4d_files[0]))
                shape_4d = img_4d.shape
                header = img_4d.header
                zooms = header.get_zooms()
                if len(shape_4d) == 4:
                    volume_shape = shape_4d[:3]
                    num_temporal_frames = shape_4d[3]
                else:
                    volume_shape = shape_4d
                    num_temporal_frames = 1
                spacing = zooms[:3]
                
                fdata = img_4d.get_fdata()
                if np.isnan(fdata).any() or np.isinf(fdata).any():
                    is_corrupted = True
                    corruption_error = "NaN or Inf values in 4D volume"
            except Exception as e:
                is_corrupted = True
                corruption_error = f"Failed to load 4D volume: {str(e)}"
        
        has_ed_img = False
        has_ed_gt = False
        ed_gt_labels = []
        if ed_frame is not None:
            ed_img_file = pdir / f"{pid}_frame{int(ed_frame):02d}.nii.gz"
            ed_gt_file = pdir / f"{pid}_frame{int(ed_frame):02d}_gt.nii.gz"
            has_ed_img = ed_img_file.exists()
            has_ed_gt = ed_gt_file.exists()
            if has_ed_gt and load_data:
                try:
                    import nibabel as nib
                    import numpy as np

                    gt_nii = nib.load(str(ed_gt_file))
                    gt_arr = gt_nii.get_fdata().astype(np.int64)
                    ed_gt_labels = [int(x) for x in np.unique(gt_arr)]
                except Exception as e:
                    is_corrupted = True
                    corruption_error += f"; Failed loading ED GT: {str(e)}"
        
        has_es_img = False
        has_es_gt = False
        es_gt_labels = []
        if es_frame is not None:
            es_img_file = pdir / f"{pid}_frame{int(es_frame):02d}.nii.gz"
            es_gt_file = pdir / f"{pid}_frame{int(es_frame):02d}_gt.nii.gz"
            has_es_img = es_img_file.exists()
            has_es_gt = es_gt_file.exists()
            if has_es_gt and load_data:
                try:
                    import nibabel as nib
                    import numpy as np

                    gt_nii = nib.load(str(es_gt_file))
                    gt_arr = gt_nii.get_fdata().astype(np.int64)
                    es_gt_labels = [int(x) for x in np.unique(gt_arr)]
                except Exception as e:
                    is_corrupted = True
                    corruption_error += f"; Failed loading ES GT: {str(e)}"
        
        return {
            "patient_id": pid,
            "pathology_group": group,
            "height_cm": height,
            "weight_kg": weight,
            "ed_frame": ed_frame,
            "es_frame": es_frame,
            "num_temporal_frames": num_temporal_frames,
            "num_labeled_frames": int(has_ed_gt) + int(has_es_gt),
            "num_unlabeled_frames": max(0, num_temporal_frames - (int(has_ed_gt) + int(has_es_gt))),
            "volume_shape_x": volume_shape[0],
            "volume_shape_y": volume_shape[1],
            "volume_shape_z": volume_shape[2],
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "has_4d_cine": has_4d,
            "has_ed_image": has_ed_img,
            "has_ed_gt": has_ed_gt,
            "has_es_image": has_es_img,
            "has_es_gt": has_es_gt,
            "ed_gt_labels": str(ed_gt_labels),
            "es_gt_labels": str(es_gt_labels),
            "is_corrupted": is_corrupted,
            "corruption_error": corruption_error,
            "file_format": "NIfTI (.nii.gz)"
        }

    def build_inventory_records(self, subset: str = "training") -> List[Dict[str, Any]]:
        """Inspect all patients in a subset and return a list of dictionaries."""
        patient_dirs = self.get_patient_dirs(subset=subset)
        records = []
        for pdir in patient_dirs:
            rec = self.inspect_patient(pdir, load_data=True)
            rec["subset"] = subset
            records.append(rec)
        return records

    def save_inventory(self, output_path: str = "data/processed/dataset_inventory.csv", subset: str = "training") -> Optional[List[Dict[str, Any]]]:
        """Inspect and save machine-readable inventory CSV."""
        is_present, msg = self.verify_presence()
        if not is_present:
            print(f"Cannot save inventory: dataset not present. {msg}")
            return None
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        records = self.build_inventory_records(subset=subset)
        if not records:
            return None
            
        fieldnames = list(records[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
            
        print(f"Inventory saved successfully ({len(records)} patients) -> {output_path}")
        return records


def main():
    inspector = ACDCInspector()
    is_present, msg = inspector.verify_presence()
    print("=" * 70)
    print("ACDC DATASET INSPECTION")
    print("=" * 70)
    print(f"Dataset Presence Status: {'AVAILABLE' if is_present else 'NOT FOUND'}")
    print(msg)
    
    if is_present:
        inventory_path = "data/processed/dataset_inventory.csv"
        records = inspector.save_inventory(inventory_path)
        if records:
            print(f"\nDataset Summary: {len(records)} patients inspected.")
    else:
        print("\nACTION REQUIRED:")
        print("Please place the ACDC dataset into 'data/raw/ACDC' as detailed above.")


if __name__ == "__main__":
    main()
