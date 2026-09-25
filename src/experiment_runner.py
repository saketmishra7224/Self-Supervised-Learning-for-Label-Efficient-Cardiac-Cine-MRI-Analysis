"""Run reproducible, patient-isolated limited-label fine-tuning experiments.

SSL and motion checkpoints are dependencies, not training steps in this module.
Each fine-tuning run has its own recoverable state and results directory.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import ACDCSegDataset, ACDCTemporalDataset, get_train_transforms, get_val_transforms
from src.encoder import count_parameters, load_encoder_weights
from src.losses import DiceCELoss
from src.motion import MotionEstimator
from src.segmentation_model import SegmentationUNet
from src.train import Trainer, compute_val_metrics_wrapper


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    return torch.device("cuda" if preference == "auto" and torch.cuda.is_available() else "cpu" if preference == "auto" else preference)


def _read_patient_ids(split_file: Union[str, Path]) -> set:
    path = Path(split_file)
    if path.suffix == ".txt":
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data if isinstance(data, list) else data.get("patients", data.get("train", [])))


def _cycled_batches(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield batches from a loader indefinitely without caching them.

    Unlike itertools.cycle (which retains every yielded batch), this
    re-iterates the loader on exhaustion so only one batch is live at a time.
    """
    while True:
        for batch in loader:
            yield batch


def validate_resume_checkpoint(resume_path: Union[str, Path], run_id: str) -> None:
    """Fail closed on cross-experiment resume before any state is restored."""
    checkpoint = torch.load(str(resume_path), map_location="cpu", weights_only=False)
    provenance = checkpoint.get("experiment_name")
    if provenance is None:
        print(f"Warning: resume checkpoint {resume_path} carries no experiment "
              f"provenance (legacy format); proceeding for run '{run_id}' "
              "on operator responsibility.")
    elif str(provenance) != run_id:
        raise ValueError(
            f"Refusing cross-experiment resume: checkpoint {resume_path} "
            f"belongs to experiment '{provenance}', not '{run_id}'."
        )


class ExperimentRegistry:
    def __init__(self, registry_file: Union[str, Path]):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.records = json.loads(self.registry_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.records = {}

    def register(self, experiment_id: str, data: Dict[str, Any]) -> None:
        self.records[experiment_id] = {**data, "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.registry_file.write_text(json.dumps(self.records, indent=2), encoding="utf-8")


class PseudoLabelArtifactDataset(Dataset):
    """Load confidence-filtered pseudo labels after validating their patients."""
    def __init__(self, processed_dir: str, manifest_path: Union[str, Path], train_patients: set, transform=None):
        self.processed_dir, self.manifest_path = Path(processed_dir), Path(manifest_path)
        self.root = self.manifest_path.parent
        self.records = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not self.records:
            raise ValueError(f"Pseudo-label manifest is empty: {self.manifest_path}")
        invalid = {str(record["patient_id"]) for record in self.records} - train_patients
        if invalid:
            raise ValueError(f"Pseudo-label manifest contains non-training patients: {sorted(invalid)}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        with np.load(self.root / record["file"], allow_pickle=True) as labels:
            patient_id = str(labels["patient_id"])
            frame_idx, slice_idx = int(labels["frame_idx"]), int(labels["slice_idx"])
            filtered = labels["filtered_pseudo_label"].astype(np.int64)
        image_path = self.processed_dir / f"{patient_id}_frame{frame_idx:02d}_slice{slice_idx:02d}.npz"
        with np.load(image_path, allow_pickle=True) as image_data:
            image = image_data["image"].astype(np.float32)
        # Existing segmentation transforms operate on image/mask pairs, which
        # keeps image and pseudo target spatially aligned.
        sample = {"image": torch.from_numpy(image).unsqueeze(0), "mask": torch.from_numpy(filtered).long()}
        if self.transform:
            sample = self.transform(sample)
        return {"image": sample["image"], "pseudo_label": sample["mask"], "patient_id": patient_id}


def _load_motion_checkpoint(model: MotionEstimator, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def build_experiment_model(config: dict, mode: str, ssl_checkpoint: Optional[str], device: torch.device) -> Tuple[SegmentationUNet, Optional[MotionEstimator]]:
    model_cfg, variant = config["model"], config["variants"][mode]
    model = SegmentationUNet(in_channels=model_cfg.get("in_channels", 1), num_classes=model_cfg.get("num_classes", 4), encoder_channels=model_cfg.get("encoder_channels", [32, 64, 128, 256]), dropout=model_cfg.get("dropout", 0.1), use_residual=model_cfg.get("use_residual", True)).to(device)
    if variant.get("use_ssl_pretrained", False):
        path = Path(ssl_checkpoint or variant["ssl_checkpoint"])
        if not path.exists():
            raise FileNotFoundError(f"{mode} requires an SSL checkpoint, but it is missing: {path}")
        load_encoder_weights(model.encoder, str(path), strict=False)
    motion = None
    if variant.get("use_motion", False):
        path = Path(variant["motion_checkpoint"])
        if not path.exists():
            raise FileNotFoundError(f"{mode} requires a motion checkpoint, but it is missing: {path}")
        motion = MotionEstimator(channels=[16, 32, 64, 32]).to(device)
        _load_motion_checkpoint(motion, path, device)
    return model, motion


def load_pseudo_dataset(config: dict, label_fraction: int, train_patients: set, transform) -> PseudoLabelArtifactDataset:
    pseudo_root = Path(config["logging"].get("pseudo_labels_dir", "results/pseudo_labels")) / f"{label_fraction}pct"
    metadata_path, manifest_path = pseudo_root / "pseudo_label_metadata.json", pseudo_root / "pseudo_label_index.json"
    if not metadata_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Pseudo-label artifacts for {label_fraction}% are required at {pseudo_root}. Run src/pseudo_labels.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("label_fraction", -1)) != label_fraction or not metadata.get("teacher_checkpoint"):
        raise ValueError("Pseudo-label metadata must record the requested fraction and its teacher checkpoint.")
    return PseudoLabelArtifactDataset(config["data"]["processed_dir"], manifest_path, train_patients, transform)


class FineTuneExperimentTrainer(Trainer):
    """Existing resumable trainer extended with train-only auxiliary losses."""
    def __init__(self, *args, motion_estimator=None, temporal_loader=None, pseudo_loader=None, motion_weight=0.0, pseudo_weight=0.0, freeze_encoder_epochs=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_estimator, self.temporal_loader, self.pseudo_loader = motion_estimator, temporal_loader, pseudo_loader
        self.motion_weight, self.pseudo_weight, self.freeze_encoder_epochs = motion_weight, pseudo_weight, freeze_encoder_epochs

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad_(epoch > self.freeze_encoder_epochs)
        self.model.train()
        if self.motion_estimator is not None:
            self.motion_estimator.eval()
        temporal_batches = _cycled_batches(self.temporal_loader) if self.temporal_loader is not None else None
        pseudo_batches = _cycled_batches(self.pseudo_loader) if self.pseudo_loader is not None else None
        totals = {"train_loss": 0.0, "motion_loss": 0.0, "pseudo_loss": 0.0}
        for batch in train_loader:
            images, masks = batch["image"].to(self.device), batch["mask"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.mixed_precision):
                total_loss = self.criterion(self.model(images), masks)
                motion_loss = pseudo_loss = None
                if temporal_batches is not None:
                    temporal = next(temporal_batches)
                    frame_t, frame_t1 = temporal["frame_t"].to(self.device), temporal["frame_t1"].to(self.device)
                    with torch.no_grad():
                        flow = self.motion_estimator(frame_t, frame_t1)["flow"]
                    probs_t, probs_t1 = F.softmax(self.model(frame_t), dim=1), F.softmax(self.model(frame_t1), dim=1)
                    motion_loss = F.mse_loss(self.motion_estimator.transformer(probs_t, flow), probs_t1)
                    total_loss = total_loss + self.motion_weight * motion_loss
                if pseudo_batches is not None:
                    pseudo = next(pseudo_batches)
                    pseudo_targets = pseudo["pseudo_label"].to(self.device)
                    if (pseudo_targets >= 0).any():
                        pseudo_loss = F.cross_entropy(self.model(pseudo["image"].to(self.device)), pseudo_targets, ignore_index=-1)
                        total_loss = total_loss + self.pseudo_weight * pseudo_loss
                if self.scaler is not None:
                    self.scaler.scale(total_loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
                else:
                    total_loss.backward(); self.optimizer.step()
            totals["train_loss"] += float(total_loss.detach())
            totals["motion_loss"] += float(motion_loss.detach()) if motion_loss is not None else 0.0
            totals["pseudo_loss"] += float(pseudo_loss.detach()) if pseudo_loss is not None else 0.0
        return {key: value / max(len(train_loader), 1) for key, value in totals.items()}


def run_experiment(config: dict, mode: str, label_fraction: int, seed: int, device: torch.device, ssl_checkpoint=None, resume=None, run_id=None, epochs_override=None, batch_size_override=None, smoke_test=False) -> Dict[str, Any]:
    variant, data_cfg, opt_cfg, loss_cfg, log_cfg = config["variants"][mode], config["data"], config["optimization"], config["loss"], config["logging"]
    train_split, val_split = data_cfg["subsets"][label_fraction], data_cfg["val_split"]
    train_patients, val_patients = _read_patient_ids(train_split), _read_patient_ids(val_split)
    if train_patients & val_patients:
        raise ValueError("Configured labeled train and validation patient splits overlap.")
    train_dataset = ACDCSegDataset(data_cfg["processed_dir"], train_split, transform=get_train_transforms())
    val_dataset = ACDCSegDataset(data_cfg["processed_dir"], val_split, transform=get_val_transforms())
    if set(train_dataset.get_patient_ids()) & set(val_dataset.get_patient_ids()):
        raise ValueError("Dataset indexing found train/validation patient leakage.")
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Fine-tuning requires non-empty labeled train and validation datasets.")
    if smoke_test:
        # Exercise the exact training/validation/checkpoint path on one batch
        # per split without turning a local CPU verification into an experiment.
        train_dataset = Subset(train_dataset, range(min(2, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(2, len(val_dataset))))
    batch_size, epochs = batch_size_override or opt_cfg.get("batch_size", 8), epochs_override or opt_cfg.get("epochs", 200)
    loader_args = {"batch_size": batch_size, "num_workers": data_cfg.get("num_workers", 0), "pin_memory": device.type == "cuda"}
    train_loader, val_loader = DataLoader(train_dataset, shuffle=True, **loader_args), DataLoader(val_dataset, shuffle=False, **loader_args)
    model, motion = build_experiment_model(config, mode, ssl_checkpoint, device)
    temporal_loader = None
    if motion is not None:
        temporal = ACDCTemporalDataset(data_cfg["processed_dir"], data_cfg["subsets"][100])
        if not len(temporal):
            raise ValueError("Motion regularization requires temporal pairs from training patients.")
        temporal_loader = DataLoader(temporal, shuffle=True, **loader_args)
    pseudo_loader = None
    if variant.get("use_pseudo_labels", False):
        pseudo_loader = DataLoader(load_pseudo_dataset(config, label_fraction, _read_patient_ids(data_cfg["subsets"][100]), get_train_transforms()), shuffle=True, **loader_args)
    run_id = run_id or f"{mode}_{label_fraction}pct_seed{seed}"
    checkpoint_dir, results_dir = Path(log_cfg["checkpoint_dir"]) / run_id, Path(log_cfg["results_dir"]) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW([{"params": model.decoder.parameters(), "lr": float(opt_cfg.get("lr_decoder", 1e-4))}, {"params": model.encoder.parameters(), "lr": float(opt_cfg.get("lr_encoder", 1e-5))}], weight_decay=float(opt_cfg.get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if opt_cfg.get("scheduler", "cosine") == "cosine" else None
    criterion = DiceCELoss(num_classes=data_cfg.get("num_classes", 4), dice_weight=loss_cfg.get("dice_weight", 1.0), ce_weight=loss_cfg.get("ce_weight", 1.0), include_background=loss_cfg.get("include_background", False))
    trainer = FineTuneExperimentTrainer(model=model, optimizer=optimizer, criterion=criterion, device=device, scheduler=scheduler, mixed_precision=opt_cfg.get("mixed_precision", True), checkpoint_dir=str(checkpoint_dir), log_dir=str(results_dir / "logs"), experiment_name=run_id, motion_estimator=motion, temporal_loader=temporal_loader, pseudo_loader=pseudo_loader, motion_weight=float(loss_cfg.get("motion_weight", 0.1)), pseudo_weight=float(loss_cfg.get("pseudo_weight", 0.25)), freeze_encoder_epochs=int(variant.get("freeze_encoder_epochs", 0)))
    if resume:
        validate_resume_checkpoint(resume, run_id)
        trainer.load_checkpoint(resume)
    metadata = {"run_id": run_id, "mode": mode, "label_fraction": label_fraction, "seed": seed, "device": str(device), "train_patients": sorted(train_patients), "validation_patients": sorted(val_patients), "ssl_checkpoint": ssl_checkpoint or variant.get("ssl_checkpoint"), "motion_checkpoint": variant.get("motion_checkpoint"), "uses_pseudo_labels": variant.get("use_pseudo_labels", False), "pseudo_label_source": str(Path(log_cfg.get("pseudo_labels_dir", "results/pseudo_labels")) / f"{label_fraction}pct") if variant.get("use_pseudo_labels", False) else None, "robustness": {"status": "not_run", "config": config.get("robustness", {})}, "parameters": count_parameters(model)}
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    trainer.train(train_loader, val_loader, n_epochs=epochs, early_stopping_patience=int(opt_cfg.get("early_stopping_patience", 30)), compute_metrics_fn=compute_val_metrics_wrapper, monitor_metric="Mean_Dice")
    latest_checkpoint = checkpoint_dir / f"{run_id}_latest.pth"
    summary = {**metadata, "status": "complete", "best_metric": trainer.best_metric, "best_epoch": trainer.best_epoch, "history_file": str(checkpoint_dir / f"{run_id}_history.json"), "benchmark": {"last_epoch_seconds": trainer.history["epoch_seconds"][-1], "last_train_samples_per_second": trainer.history["train_samples_per_second"][-1], "peak_vram_mb": max(trainer.history["peak_vram_mb"]), "peak_ram_mb": max(value for value in trainer.history["peak_ram_mb"] if value is not None) if any(value is not None for value in trainer.history["peak_ram_mb"]) else None, "latest_checkpoint_mb": latest_checkpoint.stat().st_size / (1024 ** 2) if latest_checkpoint.exists() else None}}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ExperimentRegistry(log_cfg["registry_file"]).register(run_id, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-isolated limited-label fine-tuning runner")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--mode", default="supervised", choices=["supervised", "ssl_finetune", "ssl_motion", "ssl_pseudo", "full_pipeline", "ssl_motion_pseudo"])
    parser.add_argument("--label-fraction", type=int, default=100, choices=[10, 25, 50, 100])
    parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ssl-checkpoint", default=None); parser.add_argument("--resume", default=None)
    parser.add_argument("--run-id", default=None); parser.add_argument("--epochs", type=int, default=None); parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run one CPU train/validation batch and save isolated smoke artifacts")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(config["project"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cpu") if args.smoke_test else get_device(args.device)
    run_id = args.run_id or (f"{args.mode}_{args.label_fraction}pct_seed{seed}_smoke" if args.smoke_test else None)
    epochs = 1 if args.smoke_test else args.epochs
    batch_size = 2 if args.smoke_test else args.batch_size
    print(json.dumps(run_experiment(config, args.mode, args.label_fraction, seed, device, args.ssl_checkpoint, args.resume, run_id, epochs, batch_size, args.smoke_test), indent=2))


if __name__ == "__main__":
    main()
