"""
Dataset classes for ACDC Cardiac Cine MRI.

Provides:
- ACDCRawDataset: Loads raw NIfTI files with patient metadata
- ACDCProcessedDataset: Loads preprocessed 2D slices for segmentation
- ACDCTemporalDataset: Returns temporal frame pairs for SSL pretraining
- ACDCSegDataset: Returns image-mask pairs for supervised training

All datasets preserve patient_id to enable patient-level evaluation and prevent data leakage.
"""

import os
import json
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


# =============================================================================
# ACDC label remapping: ACDC uses 0=BG, 1=RV, 2=Myo, 3=LV
# Project spec uses:          0=BG, 1=LV, 2=Myo, 3=RV
# =============================================================================
ACDC_LABEL_REMAP = {0: 0, 1: 3, 2: 2, 3: 1}


def remap_labels(mask: np.ndarray) -> np.ndarray:
    """Remap ACDC labels to project specification ordering."""
    remapped = np.zeros_like(mask)
    for src, dst in ACDC_LABEL_REMAP.items():
        remapped[mask == src] = dst
    return remapped


def parse_info_cfg(info_path: str) -> Dict:
    """Parse ACDC Info.cfg file for patient metadata."""
    info = {}
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                # Try to convert to int/float
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                info[key] = val
    return info


def resample_slice(
    image: np.ndarray,
    original_spacing: Tuple[float, float],
    target_spacing: Tuple[float, float] = (1.5, 1.5),
    is_mask: bool = False
) -> np.ndarray:
    """
    Resample a 2D slice or mask to target in-plane voxel spacing.
    Uses spline interpolation (order 3, bicubic) for images to preserve continuous gradient details.
    Uses nearest-neighbor interpolation (order 0) for segmentation masks to strictly guarantee
    discrete integer class labels {0, 1, 2, 3}.
    """
    zoom_factors = (
        original_spacing[0] / target_spacing[0],
        original_spacing[1] / target_spacing[1],
    )
    if is_mask:
        return zoom(image, zoom_factors, order=0, mode='nearest')
    else:
        return zoom(image, zoom_factors, order=3, mode='nearest')


def center_crop_or_pad(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Center-crop or zero-pad a 2D array to target spatial dimensions (target_H, target_W).
    If the image is larger than target_size, crops the centered bounding window.
    If smaller, symmetrically pads with pad_value.
    """
    h, w = image.shape
    th, tw = target_size
    result = np.full((th, tw), pad_value, dtype=image.dtype)
    
    sh = max(0, (h - th) // 2)
    sw = max(0, (w - tw) // 2)
    dh = max(0, (th - h) // 2)
    dw = max(0, (tw - w) // 2)
    
    copy_h = min(h, th)
    copy_w = min(w, tw)
    
    result[dh:dh+copy_h, dw:dw+copy_w] = image[sh:sh+copy_h, sw:sw+copy_w]
    return result


def normalize_intensity(volume: np.ndarray, method: str = "zscore", epsilon: float = 1e-8) -> np.ndarray:
    """
    Per-volume MRI intensity normalization to avoid cross-patient / cross-split data leakage.
    - 'zscore': (volume - mean) / (std + epsilon)
    - 'minmax': (volume - min) / (max - min + epsilon)
    """
    if method == "zscore":
        mean = np.mean(volume)
        std = np.std(volume)
        if std < epsilon:
            return volume - mean
        return (volume - mean) / std
    elif method == "minmax":
        vmin = np.min(volume)
        vmax = np.max(volume)
        if vmax - vmin < epsilon:
            return np.zeros_like(volume)
        return (volume - vmin) / (vmax - vmin)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_single_patient(
    patient_id: str,
    patient_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (256, 256),
    target_spacing: Tuple[float, float] = (1.5, 1.5),
    normalization: str = "zscore"
) -> Dict[str, Any]:
    """
    Preprocess all 2D cine frames and labeled ED/ES slices for a single patient.
    Extracts and saves each 2D/2D+t sample as an individual compressed .npz archive.
    """
    info_file = patient_dir / "Info.cfg"
    info = parse_info_cfg(str(info_file)) if info_file.exists() else {}
    ed = info.get("ED")
    es = info.get("ES")
    group = info.get("Group", "Unknown")
    
    # Load 4D cine volume
    cine_files = list(patient_dir.glob(f"{patient_id}_4d.nii.gz")) or list(patient_dir.glob("*_4d.nii.gz"))
    if not cine_files:
        raise FileNotFoundError(f"Missing 4D cine for {patient_id}")
    
    nii_4d = nib.load(str(cine_files[0]))
    vol_4d = nii_4d.get_fdata().astype(np.float32)
    header = nii_4d.header
    pixdim = header.get_zooms()
    original_spacing = (float(pixdim[0]), float(pixdim[1]))
    
    n_slices = vol_4d.shape[2]
    n_frames = vol_4d.shape[3] if len(vol_4d.shape) == 4 else 1
    
    # Normalize per-volume to prevent data leakage across patients
    vol_4d_norm = normalize_intensity(vol_4d, method=normalization)
    
    # Load ground truth for ED and ES
    gt_masks = {}
    for phase_name, frame_idx in [("ED", ed), ("ES", es)]:
        if frame_idx is not None:
            gt_files = list(patient_dir.glob(f"*_frame{int(frame_idx):02d}_gt.nii.gz"))
            if gt_files:
                raw_gt = nib.load(str(gt_files[0])).get_fdata().astype(np.int64)
                gt_masks[int(frame_idx)] = remap_labels(raw_gt)
    
    saved_files = []
    labeled_count = 0
    unlabeled_count = 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for f_idx in range(n_frames):
        for s_idx in range(n_slices):
            img_slice = vol_4d_norm[:, :, s_idx, f_idx]
            img_resampled = resample_slice(img_slice, original_spacing, target_spacing, is_mask=False)
            img_final = center_crop_or_pad(img_resampled, target_size, pad_value=0.0)
            
            has_gt = f_idx in gt_masks
            if has_gt:
                mask_slice = gt_masks[f_idx][:, :, s_idx]
                mask_resampled = resample_slice(mask_slice.astype(np.float32), original_spacing, target_spacing, is_mask=True)
                mask_final = center_crop_or_pad(mask_resampled.astype(np.int64), target_size, pad_value=0)
                labeled_count += 1
            else:
                mask_final = np.full(target_size, -1, dtype=np.int64)
                unlabeled_count += 1
            
            if f_idx == ed:
                phase_tag = "ED"
            elif f_idx == es:
                phase_tag = "ES"
            else:
                phase_tag = "cine"
            
            fname = f"{patient_id}_frame{f_idx:02d}_slice{s_idx:02d}.npz"
            fpath = output_dir / fname
            np.savez_compressed(
                str(fpath),
                image=img_final.astype(np.float32),
                mask=mask_final.astype(np.int64),
                patient_id=patient_id,
                slice_idx=s_idx,
                frame_idx=f_idx,
                phase=phase_tag,
                pathology=group,
                original_shape=np.array(vol_4d.shape[:2]),
                original_spacing=np.array(original_spacing)
            )
            saved_files.append(fname)
            
    return {
        "patient_id": patient_id,
        "n_slices": n_slices,
        "n_frames": n_frames,
        "n_labeled": labeled_count,
        "n_unlabeled": unlabeled_count,
        "total_files": len(saved_files)
    }



class ACDCRawDataset:
    """
    Loads raw ACDC NIfTI data with full patient metadata.
    
    This is NOT a PyTorch Dataset — it's a utility for data exploration
    and preprocessing. It inventories the dataset and provides access
    to raw volumes and metadata.
    
    Args:
        root_dir: Path to ACDC directory (containing training/ and/or testing/)
        subset: "training" or "testing"
    """
    
    def __init__(self, root_dir: str, subset: str = "training"):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.subset_dir = self.root_dir / subset
        
        if not self.subset_dir.exists():
            raise FileNotFoundError(
                f"ACDC {subset} directory not found: {self.subset_dir}\n"
                f"Please download ACDC and place it in {root_dir}"
            )
        
        # Find all patient directories
        self.patient_dirs = sorted([
            d for d in self.subset_dir.iterdir() 
            if d.is_dir() and d.name.startswith("patient")
        ])
        self.patient_ids = [d.name for d in self.patient_dirs]
        
        # Parse metadata for all patients
        self.metadata = {}
        for pid, pdir in zip(self.patient_ids, self.patient_dirs):
            info_path = pdir / "Info.cfg"
            if info_path.exists():
                self.metadata[pid] = parse_info_cfg(str(info_path))
            else:
                self.metadata[pid] = {}
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def get_patient_info(self, patient_id: str) -> Dict:
        """Get metadata for a patient."""
        return self.metadata.get(patient_id, {})
    
    def get_pathology_groups(self) -> Dict[str, List[str]]:
        """Group patients by pathology."""
        groups = {}
        for pid, info in self.metadata.items():
            group = info.get("Group", "Unknown")
            groups.setdefault(group, []).append(pid)
        return groups
    
    def load_4d_volume(self, patient_id: str) -> Tuple[np.ndarray, nib.Nifti1Header]:
        """Load 4D cine volume for a patient."""
        pdir = self.subset_dir / patient_id
        nifti_4d = list(pdir.glob("*_4d.nii.gz"))
        if not nifti_4d:
            raise FileNotFoundError(f"No 4D volume found for {patient_id}")
        
        img = nib.load(str(nifti_4d[0]))
        return img.get_fdata(), img.header
    
    def load_frame(self, patient_id: str, frame_idx: int) -> Tuple[np.ndarray, nib.Nifti1Header]:
        """Load a specific frame (3D volume at one time point)."""
        pdir = self.subset_dir / patient_id
        frame_file = list(pdir.glob(f"*_frame{frame_idx:02d}.nii.gz"))
        # Filter out ground truth files
        frame_file = [f for f in frame_file if "_gt" not in f.name]
        if not frame_file:
            raise FileNotFoundError(f"Frame {frame_idx} not found for {patient_id}")
        
        img = nib.load(str(frame_file[0]))
        return img.get_fdata(), img.header
    
    def load_frame_gt(self, patient_id: str, frame_idx: int) -> np.ndarray:
        """Load ground truth segmentation for a specific frame."""
        pdir = self.subset_dir / patient_id
        gt_file = list(pdir.glob(f"*_frame{frame_idx:02d}_gt.nii.gz"))
        if not gt_file:
            raise FileNotFoundError(f"GT for frame {frame_idx} not found for {patient_id}")
        
        img = nib.load(str(gt_file[0]))
        mask = img.get_fdata().astype(np.int64)
        return remap_labels(mask)
    
    def get_ed_es_frames(self, patient_id: str) -> Tuple[int, int]:
        """Get ED and ES frame indices from metadata."""
        info = self.metadata[patient_id]
        ed = info.get("ED", None)
        es = info.get("ES", None)
        return ed, es
    
    def get_all_frame_indices(self, patient_id: str) -> List[int]:
        """Get all available frame indices for a patient from 4D volume."""
        pdir = self.subset_dir / patient_id
        nifti_4d = list(pdir.glob("*_4d.nii.gz"))
        if nifti_4d:
            img = nib.load(str(nifti_4d[0]))
            n_frames = img.shape[-1] if len(img.shape) == 4 else 1
            return list(range(n_frames))
        return []
    
    def inventory(self) -> Dict:
        """Full dataset inventory."""
        inv = {
            "num_patients": len(self.patient_ids),
            "patients": {},
        }
        for pid in self.patient_ids:
            info = self.metadata[pid]
            ed, es = self.get_ed_es_frames(pid)
            n_frames = len(self.get_all_frame_indices(pid))
            
            # Load one frame to get spatial info
            try:
                vol, header = self.load_frame(pid, ed if ed is not None else 0)
                shape = vol.shape
                pixdim = header.get_zooms()[:3]
            except Exception:
                shape = None
                pixdim = None
            
            inv["patients"][pid] = {
                "group": info.get("Group", "Unknown"),
                "ed_frame": ed,
                "es_frame": es,
                "num_frames": n_frames,
                "volume_shape": shape,
                "pixel_spacing": pixdim,
                "height": info.get("Height", None),
                "weight": info.get("Weight", None),
            }
        return inv


class ACDCProcessedDataset(Dataset):
    """
    PyTorch Dataset for preprocessed 2D ACDC slices.
    
    Loads .npz files created during preprocessing. Each file contains:
    - image: (H, W) float32 normalized image
    - mask: (H, W) int64 segmentation mask (may be -1 if no GT)
    - patient_id, slice_idx, frame_idx, phase, pathology
    
    Args:
        processed_dir: Path to preprocessed data directory
        split_file: Path to JSON split file (list of patient IDs)
        has_labels: If True, only load samples with valid masks
        transform: Optional transform function
        label_fraction: Fraction of patients to use (for limited-label experiments)
        seed: Random seed for label fraction sampling
    """
    
    def __init__(
        self,
        processed_dir: str,
        split_file: str,
        has_labels: bool = True,
        transform: Optional[Callable] = None,
        label_fraction: float = 1.0,
        seed: int = 42,
    ):
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.has_labels = has_labels
        
        # Load patient IDs from split file (.txt or .json)
        split_path = Path(split_file)
        if split_path.suffix.lower() == ".txt":
            with open(split_file, "r") as f:
                patient_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        else:
            with open(split_file, "r") as f:
                split_data = json.load(f)
            # split_data can be a list of patient IDs or a dict with 'patients' key
            if isinstance(split_data, list):
                patient_ids = split_data
            elif isinstance(split_data, dict):
                patient_ids = split_data.get('patients', split_data.get('train', []))
            else:
                raise ValueError(f"Unexpected split file format: {type(split_data)}")
        
        # Apply label fraction (patient-level)
        if label_fraction < 1.0:
            rng = np.random.RandomState(seed)
            n_select = max(1, int(len(patient_ids) * label_fraction))
            patient_ids = sorted(rng.choice(patient_ids, n_select, replace=False).tolist())
        
        self.patient_ids = set(patient_ids)
        
        # Collect matching samples (fast path using dataset_index.json if present)
        index_file = self.processed_dir / "dataset_index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index_entries = json.load(f)
            self.samples = []
            for item in index_entries:
                if item["patient_id"] in self.patient_ids:
                    if not has_labels or item.get("has_gt", False):
                        self.samples.append(str(self.processed_dir / item["file"]))
        else:
            self.samples = []
            for npz_file in sorted(self.processed_dir.glob("*.npz")):
                fname = npz_file.stem
                match = re.match(r"(patient\d+)_", fname)
                if match and match.group(1) in self.patient_ids:
                    self.samples.append(str(npz_file))
            if has_labels:
                labeled_samples = []
                for s in self.samples:
                    data = np.load(s, allow_pickle=True)
                    if 'mask' in data and not np.all(data['mask'] == -1):
                        labeled_samples.append(s)
                self.samples = labeled_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.samples[idx], allow_pickle=True)
        
        image = data['image'].astype(np.float32)
        
        sample = {
            'image': torch.from_numpy(image).unsqueeze(0),  # (1, H, W)
            'patient_id': str(data['patient_id']),
            'slice_idx': int(data['slice_idx']),
            'frame_idx': int(data['frame_idx']),
        }
        
        if 'mask' in data and not np.all(data['mask'] == -1):
            mask = data['mask'].astype(np.int64)
            sample['mask'] = torch.from_numpy(mask).long()  # (H, W)
        
        if 'phase' in data:
            sample['phase'] = str(data['phase'])
        if 'pathology' in data:
            sample['pathology'] = str(data['pathology'])
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ACDCTemporalDataset(Dataset):
    """
    Dataset returning pairs of temporally adjacent frames for SSL pretraining.
    
    For each sample, returns (frame_t, frame_{t+1}) from the same patient
    and same slice position. Uses ALL temporal frames (not just ED/ES).
    
    Args:
        processed_dir: Path to preprocessed data directory
        split_file: Path to JSON split file
        transform: Optional augmentation transform
    """
    
    def __init__(
        self,
        processed_dir: str,
        split_file: str,
        transform: Optional[Callable] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        
        # Load patient IDs (.txt or .json)
        split_path = Path(split_file)
        if split_path.suffix.lower() == ".txt":
            with open(split_file, "r") as f:
                patient_ids = set([line.strip() for line in f if line.strip() and not line.startswith("#")])
        else:
            with open(split_file, "r") as f:
                split_data = json.load(f)
            if isinstance(split_data, list):
                patient_ids = set(split_data)
            else:
                patient_ids = set(split_data.get('patients', split_data.get('train', [])))
        
        # Build index: (patient_id, slice_idx) -> sorted list of (frame_idx, filepath)
        # Fast path using dataset_index.json if available
        self._index = {}
        index_file = self.processed_dir / "dataset_index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index_entries = json.load(f)
            for item in index_entries:
                pid = item["patient_id"]
                if pid in patient_ids:
                    frame_idx = item["frame_idx"]
                    slice_idx = item["slice_idx"]
                    key = (pid, slice_idx)
                    self._index.setdefault(key, []).append((frame_idx, str(self.processed_dir / item["file"])))
        else:
            for npz_file in sorted(self.processed_dir.glob("*.npz")):
                fname = npz_file.stem
                match = re.match(r"(patient\d+)_frame(\d+)_slice(\d+)", fname)
                if match:
                    pid = match.group(1)
                    if pid in patient_ids:
                        frame_idx = int(match.group(2))
                        slice_idx = int(match.group(3))
                        key = (pid, slice_idx)
                        self._index.setdefault(key, []).append((frame_idx, str(npz_file)))
        
        # Sort each sequence by frame index
        for key in self._index:
            self._index[key].sort(key=lambda x: x[0])
        
        # Create pairs of adjacent frames
        self.pairs = []
        for key, frames in self._index.items():
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i][1], frames[i + 1][1]))
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path_t, path_t1 = self.pairs[idx]
        
        data_t = np.load(path_t, allow_pickle=True)
        data_t1 = np.load(path_t1, allow_pickle=True)
        
        img_t = torch.from_numpy(data_t['image'].astype(np.float32)).unsqueeze(0)
        img_t1 = torch.from_numpy(data_t1['image'].astype(np.float32)).unsqueeze(0)
        
        sample = {
            'frame_t': img_t,       # (1, H, W)
            'frame_t1': img_t1,     # (1, H, W)
            'patient_id': str(data_t['patient_id']),
            'slice_idx': int(data_t['slice_idx']),
            'frame_idx_t': int(data_t['frame_idx']),
            'frame_idx_t1': int(data_t1['frame_idx']),
        }
        
        # Ground-truth masks and indicators when available (e.g. at ED/ES phases)
        if 'mask' in data_t:
            mask_t = data_t['mask'].astype(np.int64)
            sample['mask_t'] = torch.from_numpy(mask_t).long()
            sample['has_gt_t'] = bool(not np.all(mask_t == -1))
        if 'mask' in data_t1:
            mask_t1 = data_t1['mask'].astype(np.int64)
            sample['mask_t1'] = torch.from_numpy(mask_t1).long()
            sample['has_gt_t1'] = bool(not np.all(mask_t1 == -1))
            
        if 'phase' in data_t:
            sample['phase_t'] = str(data_t['phase'])
        if 'phase' in data_t1:
            sample['phase_t1'] = str(data_t1['phase'])
        if 'pathology' in data_t:
            sample['pathology'] = str(data_t['pathology'])
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ACDCSegDataset(Dataset):
    """
    Segmentation dataset that returns (image, mask) pairs.
    
    Convenience wrapper around ACDCProcessedDataset that ensures
    all samples have valid segmentation masks. Used for supervised
    training and evaluation.
    
    Args:
        processed_dir: Path to preprocessed data directory
        split_file: Path to JSON split file
        transform: Optional augmentation transform
        label_fraction: Fraction of patients with labels to use
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        processed_dir: str,
        split_file: str,
        transform: Optional[Callable] = None,
        label_fraction: float = 1.0,
        seed: int = 42,
    ):
        self.inner = ACDCProcessedDataset(
            processed_dir=processed_dir,
            split_file=split_file,
            has_labels=True,
            transform=None,
            label_fraction=label_fraction,
            seed=seed,
        )
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.inner)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.inner[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_patient_ids(self) -> List[str]:
        """Return list of unique patient IDs in this dataset."""
        return sorted(self.inner.patient_ids)


def get_train_transforms(input_size: Tuple[int, int] = (256, 256)):
    """Standard training augmentations for cardiac MRI segmentation."""
    import torchvision.transforms.functional as TF
    import random
    
    def transform(sample):
        image = sample['image']   # (1, H, W)
        mask = sample.get('mask', None)  # (H, W)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        
        # Random rotation (±15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            if mask is not None:
                mask = TF.rotate(mask.unsqueeze(0), angle, 
                                interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        
        # Random intensity shift
        if random.random() > 0.5:
            shift = random.uniform(-0.1, 0.1)
            image = image + shift
        
        # Random intensity scale
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            image = image * scale
        
        sample['image'] = image
        if mask is not None:
            sample['mask'] = mask
        
        return sample
    
    return transform


def get_val_transforms():
    """Validation transforms (identity — no augmentation)."""
    def transform(sample):
        return sample
    return transform
