"""Qualitative-only test-set visualization for Full Pipeline runs.

Inference-only script: loads the existing production *_best.pth checkpoints,
runs a single forward pass on a small deterministic subset of the fixed
20-patient TEST split, and saves MRI / Ground Truth / Prediction / Overlay
figures. It performs no training, no metric computation, no checkpoint
selection, and never writes to results/experiments/, results/tables/,
checkpoints/, or data/splits/.

Usage:
    python -m src.generate_qualitative_results --all --device cuda
    python -m src.generate_qualitative_results --run-id full_pipeline_10pct_seed42 --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import ACDCSegDataset, get_val_transforms
from src.evaluate_test import (
    PRODUCTION_RUN_IDS,
    build_model_from_config,
    parse_run_id,
)
from src.segmentation_model import SegmentationUNet

# Discrete mask colors: 0 background (unused/transparent), 1 LV red,
# 2 myocardium green, 3 RV blue.
MASK_CMAP = ListedColormap(["black", "#e41a1c", "#4daf4a", "#377eb8"])

# Deterministic example selection: sorted patient IDs at these positions,
# and the median foreground-bearing labeled slice per patient.
EXAMPLE_PATIENT_POSITIONS = (0, 10)


def resolve_checkpoint(run_id: str) -> Path:
    """Return the exact production checkpoint path for an allowlisted run."""
    parse_run_id(run_id)  # rejects anything outside PRODUCTION_RUN_IDS
    path = Path("checkpoints/experiments") / run_id / f"{run_id}_best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Production checkpoint not found: {path}")
    return path


def load_production_model(
    run_id: str, config: dict, device: torch.device
) -> SegmentationUNet:
    """Load a production checkpoint without modifying or resaving it."""
    checkpoint_path = resolve_checkpoint(run_id)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    provenance = checkpoint.get("experiment_name")
    if provenance is not None:
        if str(provenance) != run_id:
            raise ValueError(
                f"Refusing {checkpoint_path}: checkpoint provenance "
                f"'{provenance}' does not match requested run '{run_id}'."
            )
    else:
        print(
            f"Note: {checkpoint_path} is a legacy checkpoint without embedded "
            f"provenance; continuing only because '{run_id}' was explicitly "
            "selected from the fixed production allowlist."
        )
    model = build_model_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def select_examples(dataset: ACDCSegDataset) -> list:
    """Deterministically pick 2 patients x median foreground slice.

    Selection uses only patient/slice ordering and foreground presence —
    never test metrics or prediction quality.
    """
    patient_ids = sorted(dataset.get_patient_ids())
    chosen_pids = [patient_ids[i] for i in EXAMPLE_PATIENT_POSITIONS]
    by_patient = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        by_patient.setdefault(str(sample["patient_id"]), []).append(idx)
    examples = []
    for pid in chosen_pids:
        indices = sorted(
            by_patient[pid], key=lambda i: int(dataset[i]["slice_idx"])
        )
        fg = [
            i for i in indices if bool((dataset[i]["mask"] > 0).any())
        ]
        pool = fg if fg else indices
        examples.append(pool[len(pool) // 2])
    return examples


@torch.no_grad()
def predict_indices(
    model: SegmentationUNet, dataset: ACDCSegDataset, indices: list, device: torch.device
) -> dict:
    """Run inference for the selected sample indices (no gradients)."""
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    wanted = set(indices)
    preds = {}
    for batch_idx, batch in enumerate(loader):
        base = batch_idx * loader.batch_size
        images = batch["image"].to(device)
        logits = model(images)
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        for j in range(batch_preds.shape[0]):
            if base + j in wanted:
                preds[base + j] = batch_preds[j]
    return preds


def render_panels(fig_axes, images, gts, preds, pids, title):
    """Fill a (n_examples x 4) axes grid: MRI | GT | Pred | Overlay."""
    for row, (img, gt, pred, pid) in enumerate(zip(images, gts, preds, pids)):
        fig_axes[row][0].imshow(img, cmap="gray")
        fig_axes[row][0].set_title("MRI" if row == 0 else "")
        fig_axes[row][0].set_ylabel(f"{pid}\nslice", rotation=0, labelpad=30,
                                    va="center", fontsize=9)
        fig_axes[row][1].imshow(gt, cmap=MASK_CMAP, vmin=0, vmax=3)
        fig_axes[row][1].set_title("Ground Truth" if row == 0 else "")
        fig_axes[row][2].imshow(pred, cmap=MASK_CMAP, vmin=0, vmax=3)
        fig_axes[row][2].set_title("Prediction" if row == 0 else "")
        fig_axes[row][3].imshow(img, cmap="gray")
        fig_axes[row][3].imshow(
            np.ma.masked_where(pred == 0, pred),
            cmap=MASK_CMAP, vmin=0, vmax=3, alpha=0.55,
        )
        fig_axes[row][3].set_title("Overlay" if row == 0 else "")
        for ax in fig_axes[row]:
            ax.axis("off")
    fig_axes[0][0].figure.suptitle(title, fontsize=11)


def save_run_figure(run_id, label_fraction, images, gts, preds, pids, output_dir):
    fig, axes = plt.subplots(len(images), 4, figsize=(12, 4 * len(images)))
    render_panels(
        np.atleast_2d(axes), images, gts, preds, pids,
        f"{run_id} ({label_fraction}% labels) — TEST-set qualitative examples",
    )
    fig.tight_layout()
    path = Path(output_dir) / f"{run_id}_qualitative.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def save_comparison_figure(per_run_preds, images, gts, pids, output_dir):
    """Compact 2-row comparison across the four label fractions."""
    fractions = [10, 25, 50, 100]
    n_cols = 2 + len(fractions)
    fig, axes = plt.subplots(len(images), n_cols,
                             figsize=(3 * n_cols, 3.4 * len(images)))
    axes = np.atleast_2d(axes)
    headers = ["MRI", "Ground Truth"] + [f"Pred {f}%" for f in fractions]
    for row, (img, gt, pid) in enumerate(zip(images, gts, pids)):
        axes[row][0].imshow(img, cmap="gray")
        axes[row][1].imshow(gt, cmap=MASK_CMAP, vmin=0, vmax=3)
        for col, frac in enumerate(fractions):
            run = f"full_pipeline_{frac}pct_seed42"
            axes[row][2 + col].imshow(per_run_preds[run][row],
                                      cmap=MASK_CMAP, vmin=0, vmax=3)
        axes[row][0].set_ylabel(f"{pid}\nslice", rotation=0, labelpad=30,
                                va="center", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=10)
    fig.suptitle("Full Pipeline TEST predictions across label fractions "
                 "(same deterministic examples)", fontsize=11)
    fig.tight_layout()
    path = Path(output_dir) / "final_qualitative_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualitative TEST visualizations for Full Pipeline runs"
    )
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default="results/figures/qualitative")
    args = parser.parse_args()

    if args.all:
        run_ids = list(PRODUCTION_RUN_IDS)
    elif args.run_id is not None:
        run_ids = [args.run_id]
    else:
        parser.error("Provide --run-id <run_id> or --all.")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = config.get("data", {})
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    processed_dir = args.processed_dir or data_cfg.get("processed_dir", "data/processed")
    test_split = data_cfg.get("test_split", "data/splits/test_patients.txt")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ACDCSegDataset(processed_dir, test_split, transform=get_val_transforms())
    if len(dataset) == 0:
        raise ValueError(f"Test dataset is empty: {test_split}")
    example_indices = select_examples(dataset)
    images, gts, pids = [], [], []
    for idx in example_indices:
        sample = dataset[idx]
        images.append(sample["image"].squeeze(0).numpy())
        gts.append(sample["mask"].numpy())
        pids.append(str(sample["patient_id"]))

    per_run_preds = {}
    for run_id in tqdm(run_ids, desc="Qualitative runs"):
        label_fraction, _ = parse_run_id(run_id)
        model = load_production_model(run_id, config, device)
        preds = predict_indices(model, dataset, example_indices, device)
        ordered = [preds[i] for i in example_indices]
        per_run_preds[run_id] = ordered
        save_run_figure(run_id, label_fraction, images, gts, ordered, pids,
                        output_dir)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.all:
        save_comparison_figure(per_run_preds, images, gts, pids, output_dir)


if __name__ == "__main__":
    main()
