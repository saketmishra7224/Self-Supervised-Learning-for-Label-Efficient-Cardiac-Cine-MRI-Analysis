#!/usr/bin/env python3
"""MedSAM mask-decoder fine-tuning pilot; validation only, no test evaluation."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = Path("/kaggle/input/datasets/nishantk2310593/motion-guided-ssl-acdc-processed/data/processed")
DEFAULT_CHECKPOINT = Path("/kaggle/working/checkpoints/medsam/medsam_vit_b.pth")
DEFAULT_OUTPUT = Path("/kaggle/working/medsam_finetune")
CLASSES = (1, 2, 3)
IMAGE_SIZE = 1024
PROJECT_SIZE = 256
SEED = 42


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def load_splits():
    obj = json.loads((REPO / "data/splits/patient_splits.json").read_text())
    train, val, test = (set(map(str, obj[k])) for k in ("train", "val", "test"))
    if train & val or train & test or val & test:
        raise RuntimeError("Patient split leakage detected.")
    return train, val


def load_boxes():
    path = REPO / "configs/medsam_prior_boxes.json"
    obj = json.loads(path.read_text())
    if obj.get("metadata", {}).get("n_labeled_slices_used") != 1324:
        raise RuntimeError("Unexpected prior-box provenance.")
    boxes = {c: [float(v) for v in obj["boxes"][str(c)]] for c in CLASSES}
    for c, (x1, y1, x2, y2) in boxes.items():
        if not (0 <= x1 < x2 <= PROJECT_SIZE and 0 <= y1 < y2 <= PROJECT_SIZE):
            raise ValueError(f"Invalid prior box for class {c}: {boxes[c]}")
    return boxes


class LabeledSlices(Dataset):
    def __init__(self, data_dir, patient_ids):
        self.data_dir = Path(data_dir)
        rows = json.loads((self.data_dir / "dataset_index.json").read_text())
        self.rows = sorted(
            [r for r in rows if str(r["patient_id"]) in patient_ids and r.get("has_gt", False)],
            key=lambda r: (r["patient_id"], int(r["frame_idx"]), int(r["slice_idx"])),
        )
        if not self.rows:
            raise RuntimeError("No labeled slices found.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.data_dir / row["file"], allow_pickle=False) as z:
            image = z["image"].astype(np.float32)
            mask = z["mask"].astype(np.int64)
        if image.shape != (PROJECT_SIZE, PROJECT_SIZE):
            raise ValueError(f"Unexpected image shape: {image.shape}")
        if not set(np.unique(mask).tolist()).issubset({0, 1, 2, 3}):
            raise ValueError(f"Unexpected mask labels in {row['file']}")
        return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy()), row["file"]


def make_input(image, sam, transform, device):
    lo, hi = float(image.min()), float(image.max())
    scaled = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)
    rgb = np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=2)
    resized = transform.apply_image(np.ascontiguousarray(rgb))
    tensor = torch.as_tensor(resized, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return sam.preprocess(tensor)


def make_boxes(boxes, transform, device):
    arr = torch.tensor([boxes[c] for c in CLASSES], device=device, dtype=torch.float32)
    return transform.apply_boxes_torch(arr, (PROJECT_SIZE, PROJECT_SIZE))


def binary_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = logits.sigmoid()
    dice = 1 - (2 * (probs * target).sum() + 1e-6) / (probs.sum() + target.sum() + 1e-6)
    return bce + dice


def forward_logits(sam, embedding, boxes_1024):
    outputs = []
    for i in range(len(CLASSES)):
        with torch.no_grad():
            sparse, dense = sam.prompt_encoder(
                points=None, boxes=boxes_1024[i:i + 1], masks=None
            )
        low_res, _ = sam.mask_decoder(
            image_embeddings=embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        outputs.append(sam.postprocess_masks(
            low_res, input_size=(IMAGE_SIZE, IMAGE_SIZE),
            original_size=(PROJECT_SIZE, PROJECT_SIZE),
        ))
    return torch.cat(outputs, dim=1)


def foreground_dice(pred, target):
    values = []
    for c in CLASSES:
        p, t = pred == c, target == c
        denom = p.sum().item() + t.sum().item()
        values.append(1.0 if denom == 0 else 2 * (p & t).sum().item() / denom)
    return float(np.mean(values))


def run_epoch(sam, loader, boxes_1024, device, transform, optimizer, limit=0):
    training = optimizer is not None
    sam.mask_decoder.train(training)
    losses, dices = [], []
    started = time.time()
    for step, (images, masks, filenames) in enumerate(loader):
        if limit and step >= limit:
            break
        image = images[0].numpy()
        target = masks.to(device=device, dtype=torch.long)
        input_image = make_input(image, sam, transform, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            embedding = sam.image_encoder(input_image)
        logits = forward_logits(sam, embedding, boxes_1024)
        loss = torch.stack([
            binary_loss(logits[:, i:i+1], (target == c).float().unsqueeze(1))
            for i, c in enumerate(CLASSES)
        ]).mean()
        if training:
            loss.backward()
            grad_norm = sum(
                p.grad.detach().norm().item() ** 2
                for p in sam.mask_decoder.parameters() if p.grad is not None
            ) ** 0.5
            if not np.isfinite(grad_norm) or grad_norm == 0:
                raise RuntimeError("Mask decoder gradients are invalid.")
            optimizer.step()
        with torch.no_grad():
            pred = torch.cat([torch.zeros_like(logits[:, :1]), logits], dim=1).argmax(1)
            dices.append(foreground_dice(pred[0].cpu(), target[0].cpu()))
        losses.append(float(loss.detach().item()))
        if step % 10 == 0:
            print(f"{'train' if training else 'val'} {step + 1}/{len(loader)} loss={loss.item():.4f}", flush=True)
    if not losses:
        raise RuntimeError("No batches processed.")
    return {"loss": float(np.mean(losses)), "mean_dice": float(np.mean(dices)),
            "samples": len(losses), "seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    seed_everything()
    train_ids, val_ids = load_splits()
    boxes = load_boxes()
    train_ds = LabeledSlices(args.data, train_ids)
    val_ds = LabeledSlices(args.data, val_ids)
    if len(train_ds) != 1324 or len(val_ds) != 180:
        raise RuntimeError(f"Unexpected labeled counts: train={len(train_ds)}, val={len(val_ds)}")

    output = args.output
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    sam = sam_model_registry["vit_b"](checkpoint=None)
    sam.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    device = torch.device("cuda")
    sam.to(device).eval()
    for p in sam.parameters():
        p.requires_grad_(False)
    for p in sam.mask_decoder.parameters():
        p.requires_grad_(True)

    transform = ResizeLongestSide(IMAGE_SIZE)
    boxes_1024 = make_boxes(boxes, transform, device)
    optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=1e-5, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    config = {
        "seed": SEED, "epochs": args.epochs, "lr": 1e-5,
        "train_patients": len(train_ids), "validation_patients": len(val_ids),
        "train_labeled_slices": len(train_ds), "validation_labeled_slices": len(val_ds),
        "test_split_loaded": False, "trainable_module": "mask_decoder",
        "prior_boxes": boxes, "checkpoint": str(args.checkpoint),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    torch.save({"mask_decoder": sam.mask_decoder.state_dict(), "epoch": 0},
               output / "checkpoints/initial_decoder.pt")

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        tr = run_epoch(sam, train_loader, boxes_1024, device, transform, optimizer, args.max_train_batches)
        va = run_epoch(sam, val_loader, boxes_1024, device, transform, None, args.max_val_batches)
        record = {"epoch": epoch, "train": tr, "validation": va}
        history.append(record)
        (output / "logs/history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record, indent=2), flush=True)
        if va["mean_dice"] > best:
            best = va["mean_dice"]
            torch.save({
                "mask_decoder": sam.mask_decoder.state_dict(),
                "epoch": epoch, "validation_mean_dice": best,
                "optimizer": optimizer.state_dict(), "config": config,
            }, output / "checkpoints/best_decoder.pt")
            print(f"Saved best decoder: validation Dice={best:.6f}", flush=True)
    print("Pilot completed:", output)


if __name__ == "__main__":
    main()
