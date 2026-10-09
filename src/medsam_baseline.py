"""MedSAM zero-shot baseline (bowang-lab/MedSAM).

Official MedSAM checkpoint (``medsam_vit_b.pth``): SAM ViT-B image encoder +
SAM prompt encoder + SAM mask decoder, finetuned on medical images
(Apache-2.0, research use). Used here strictly zero-shot: all weights frozen,
no ACDC training or fine-tuning of any kind.

Prompt-leakage design (enforced, not assumed):
- Box prompts come exclusively from PriorBoxPrompts, a population prior
  fitted on TRAINING patients' ground truth only
  (see fit_prior_boxes). The same three fixed boxes are used for every
  validation/test image, so test-time prompts are provably independent of
  test ground-truth masks.
- Ground-truth masks from training patients may be used to derive training
  prompts (i.e. to fit the prior). Validation/test masks are never read for
  prompting, tuning, or assignment.
- No prompt/threshold/assignment tuning on the test set. Fixed rules below.

Four-class mapping (matches project convention):
- MedSAM is run once per foreground class (LV=1, MYO=2, RV=3) with that
  class's prior box; each run yields a logit map. Per-pixel class is
  argmax over the three class logits; pixels where all three logits are
  negative become background (0), mirroring SAM's 0.0 mask threshold.
  Empty predictions (e.g. absent structure) therefore yield background,
  never a hallucinated class.

Preprocessing (documented adaptation, dataset untouched):
- Project images are z-scored float32 (1, H, W). Each slice is min-max
  scaled to [0, 255] uint8 and channel-replicated to RGB for SAM's
  ResizeLongestSide(1024) + ImageNet-norm pipeline (handled internally by
  SamPredictor.set_image).
- Prior boxes live in 256 coordinates and are scaled x4 to MedSAM space.
- Logits are restored to 256x256 before merging.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = (1, 2, 3)  # LV, myocardium, RV
CLASS_NAMES = {1: "LV", 2: "myocardium", 3: "RV"}
PRIOR_LO_PCT = 5.0
PRIOR_HI_PCT = 95.0
MEDSAM_IMAGE_SIZE = 1024
PROJECT_IMAGE_SIZE = 256
COORD_SCALE = MEDSAM_IMAGE_SIZE / PROJECT_IMAGE_SIZE  # 4.0

DEFAULT_CHECKPOINT_ENV = "MEDSAM_CHECKPOINT"


def default_checkpoint_path() -> Path:
    """Expected location of the official medsam_vit_b.pth weights."""
    return Path.home() / ".cache" / "medsam" / "MedSAM" / "medsam_vit_b.pth"


def resolve_checkpoint_path(explicit: Optional[str] = None) -> Path:
    """Resolve weights path or fail clearly (never silent random init)."""
    import os

    if explicit:
        path = Path(explicit)
    elif os.environ.get(DEFAULT_CHECKPOINT_ENV):
        path = Path(os.environ[DEFAULT_CHECKPOINT_ENV])
    else:
        path = default_checkpoint_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Official MedSAM checkpoint not found: {path}. "
            "Download medsam_vit_b.pth from the official bowang-lab/MedSAM "
            "release (Google Drive folder 1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) "
            f"or set ${DEFAULT_CHECKPOINT_ENV}."
        )
    return path


def fit_prior_boxes(
    processed_dir: str,
    split_file: str,
    out_json: Optional[str] = None,
) -> Dict[int, List[float]]:
    """Fit per-class population box priors from TRAINING patients only.

    For each foreground class, collects tight boxes over labeled training
    slices and stores the 5th percentile of minima / 95th percentile of
    maxima (256 coords, [x_min, y_min, x_max, y_max]). Deterministic.
    """
    with open(split_file) as f:
        train_ids = sorted({line.strip() for line in f if line.strip()})
    boxes: Dict[int, List[List[float]]] = {c: [] for c in CLASSES}
    n_slices = 0
    for npz_path in sorted(Path(processed_dir).glob("patient*_frame*_slice*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            if str(data["patient_id"]) not in set(train_ids):
                continue
            mask = np.asarray(data["mask"])
            if (mask == -1).all():
                continue
            n_slices += 1
            for c in CLASSES:
                ys, xs = np.nonzero(mask == c)
                if len(xs) == 0:
                    continue
                boxes[c].append([float(xs.min()), float(ys.min()),
                                 float(xs.max()), float(ys.max())])
    prior = {}
    for c in CLASSES:
        if not boxes[c]:
            raise ValueError(f"No training foreground found for class {c}")
        arr = np.array(boxes[c])
        prior[c] = [float(np.percentile(arr[:, 0], PRIOR_LO_PCT)),
                    float(np.percentile(arr[:, 1], PRIOR_LO_PCT)),
                    float(np.percentile(arr[:, 2], PRIOR_HI_PCT)),
                    float(np.percentile(arr[:, 3], PRIOR_HI_PCT))]
    if out_json:
        artifact = {
            "metadata": {
                "source_split": str(split_file),
                "coordinate_convention": "x_min, y_min, x_max, y_max in 256x256 image coordinates",
                "method": f"{PRIOR_LO_PCT:.0f}th percentile of minima / {PRIOR_HI_PCT:.0f}th percentile of maxima over labeled training slices",
                "classes": {str(c): CLASS_NAMES[c] for c in CLASSES},
                "n_labeled_slices_used": n_slices,
            },
            "boxes": {str(c): v for c, v in prior.items()},
        }
        Path(out_json).write_text(json.dumps(artifact, indent=2))
    return prior


def default_prior_boxes_path() -> Path:
    """Version-controlled prior-box artifact for normal dispatch."""
    return Path(__file__).resolve().parent.parent / "configs" / "medsam_prior_boxes.json"


def load_prior_boxes(path: Optional[str] = None) -> Dict[int, List[float]]:
    """Load and validate the prior-box artifact (fail clearly, never invent).

    Raises FileNotFoundError if the artifact is absent (e.g. dataset/split
    unavailable) and ValueError on malformed, out-of-bounds, or degenerate
    boxes. Never reads validation/test labels.
    """
    artifact_path = Path(path) if path else default_prior_boxes_path()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"MedSAM prior-box artifact not found: {artifact_path}. "
            "Generate it with fit_prior_boxes() from the training split "
            "(see configs/medsam_prior_boxes.json provenance). Refusing to "
            "invent prompt boxes or read validation/test labels."
        )
    artifact = json.loads(artifact_path.read_text())
    boxes = artifact.get("boxes")
    if not isinstance(boxes, dict):
        raise ValueError(f"Prior-box artifact {artifact_path} lacks a 'boxes' mapping.")
    normalized = {int(k): [float(v) for v in vals] for k, vals in boxes.items()}
    missing = set(CLASSES) - set(normalized)
    if missing:
        raise ValueError(f"Prior-box artifact missing classes: {sorted(missing)}.")
    for c, (x1, y1, x2, y2) in normalized.items():
        values = (x1, y1, x2, y2)
        if not all(np.isfinite(values)):
            raise ValueError(f"Non-finite prior box for class {c}.")
        if not (0.0 <= x1 < x2 <= 256.0 and 0.0 <= y1 < y2 <= 256.0):
            raise ValueError(f"Out-of-bounds or degenerate prior box for class {c}: {values}.")
    return normalized


class PriorBoxPrompts(nn.Module):
    """Fixed population-prior boxes: identical for every image, GT-free at inference.

    Because prompts do not depend on the input image at all, validation/test
    prompts cannot leak ground-truth information by construction.
    """

    def __init__(self, prior_boxes: Dict[int, List[float]]):
        super().__init__()
        normalized = {int(k): [float(v) for v in vals] for k, vals in prior_boxes.items()}
        self.boxes_256 = normalized
        missing = set(CLASSES) - set(self.boxes_256)
        if missing:
            raise ValueError(f"Prior boxes missing classes: {sorted(missing)}")

    def boxes_for_image(self) -> torch.Tensor:
        """Return (3, 4) boxes in MedSAM 1024 coordinates for LV/MYO/RV."""
        rows = []
        for c in CLASSES:
            x1, y1, x2, y2 = (v * COORD_SCALE for v in self.boxes_256[c])
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(MEDSAM_IMAGE_SIZE - 1), x2), min(float(MEDSAM_IMAGE_SIZE - 1), y2)
            if not (x2 > x1 and y2 > y1):
                # Degenerate prior: deterministic full-image fallback (documented;
                # does not occur with real fitted priors, which span foreground).
                x1, y1, x2, y2 = 0.0, 0.0, float(MEDSAM_IMAGE_SIZE - 1), float(MEDSAM_IMAGE_SIZE - 1)
            rows.append([x1, y1, x2, y2])
        return torch.tensor(rows, dtype=torch.float32)


class MedSAMSegmenter(nn.Module):
    """Frozen zero-shot MedSAM wrapper returning (B, 4, H, W) pseudo-logits.

    Logit construction preserves SAM semantics for the argmax evaluator:
    channel 0 (background) is identically 0, channels 1-3 hold MedSAM
    mask logits; argmax therefore yields background exactly where all
    class logits are negative (SAM's 0.0 threshold convention).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        prompts: Optional[PriorBoxPrompts] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        from segment_anything import SamPredictor, sam_model_registry

        path = resolve_checkpoint_path(checkpoint_path)
        self.checkpoint_used = str(path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry["vit_b"]()
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        sam.load_state_dict(state, strict=True)
        sam.to(self.device)
        sam.eval()
        for p in sam.parameters():
            p.requires_grad_(False)  # zero-shot baseline: permanently frozen
        self.sam = sam  # registered submodule: parameters()/to()/state_dict() work
        self.predictor = SamPredictor(sam)
        self.prompts = prompts
        self._fitted_from_train_only = prompts is not None

    @staticmethod
    def _to_rgb_uint8(image_1chw: torch.Tensor) -> np.ndarray:
        """Min-max scale one z-scored slice to uint8 RGB (documented adaptation)."""
        img = image_1chw.detach().cpu().float().squeeze(0).numpy()
        lo, hi = float(img.min()), float(img.max())
        if hi > lo:
            img = (img - lo) / (hi - lo)
        else:
            img = np.zeros_like(img)
        rgb = np.stack([(img * 255.0).astype(np.uint8)] * 3, axis=-1)
        return np.ascontiguousarray(rgb)  # (H, W, 3)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Args: images (B, 1, H, W). Returns: (B, 4, H, W) pseudo-logits."""
        if self.prompts is None:
            raise RuntimeError("MedSAMSegmenter has no prompt generator attached.")
        b, _, h, w = images.shape
        assert (h, w) == (PROJECT_IMAGE_SIZE, PROJECT_IMAGE_SIZE), (h, w)
        boxes_1024 = self.prompts.boxes_for_image().to(self.device)
        out = torch.zeros(b, 4, h, w, dtype=torch.float32)
        for i in range(b):
            self.predictor.set_image(self._to_rgb_uint8(images[i]))
            for j in range(len(CLASSES)):
                _masks, _scores, logits = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes_1024[j : j + 1],
                    multimask_output=False,
                    return_logits=True,
                )
                logit = logits[0, 0]
                if tuple(logit.shape[-2:]) != (h, w):
                    logit = F.interpolate(
                        logit.unsqueeze(0).unsqueeze(0), size=(h, w),
                        mode="bilinear", align_corners=False).squeeze()
                out[i, CLASSES[j]] = logit.detach().cpu()
            self.predictor.reset_image()
        return out


def build_medsam(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    prior_boxes: Optional[Dict[int, List[float]]] = None,
    device: Optional[torch.device] = None,
) -> MedSAMSegmenter:
    """Build the MedSAM baseline. MedSAM has no random-init mode: with
    pretrained=False (or missing weights) this raises instead of silently
    substituting random weights."""
    if num_classes != 4 or in_channels != 1:
        raise ValueError("MedSAM baseline requires num_classes=4, in_channels=1.")
    if not pretrained:
        raise ValueError("MedSAM baseline requires the official pretrained checkpoint (zero-shot); random init is not a valid MedSAM configuration.")
    if prior_boxes is None:
        # Normal dispatch path: load the version-controlled training-only
        # prior artifact (fails clearly if absent; never invents boxes).
        prior_boxes = load_prior_boxes()
    elif isinstance(prior_boxes, (str, Path)):
        prior_boxes = load_prior_boxes(prior_boxes)
    prompts = PriorBoxPrompts(prior_boxes) if prior_boxes is not None else None
    return MedSAMSegmenter(checkpoint_path=checkpoint_path, prompts=prompts, device=device)
