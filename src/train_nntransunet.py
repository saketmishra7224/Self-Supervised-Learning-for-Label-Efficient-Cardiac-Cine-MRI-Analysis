"""nn-TransUNet training runner (paper protocol).

Dedicated entry point for the nn-TransUNet baseline (Zhao et al., Life
2022;12(10):1570). The shared ``src/train.py`` must NOT be used for this
architecture: it hardcodes AdamW + cosine/plateau scheduling + Mean_Dice
monitoring, which would silently replace the paper protocol.

Protocol honored here (locked values in ``src/nntransunet.py``):
  - SGD + Nesterov (momentum 0.99), lr 0.01, weight decay 0.0
  - PolynomialLR (power 0.9), stepped once per epoch by the shared Trainer
  - epochs x iters_per_epoch budget (paper: 500 x 250), batch fitted to GPU
  - nnU-Net v2 training augmentation (paper protocol)
  - best checkpoint by validation loss; NO early stopping

The 250-iterations-per-epoch budget is expressed with a with-replacement
``RandomSampler`` (exactly ``iters_per_epoch * batch_size`` samples per
epoch), so the shared ``Trainer`` needs no modification. Checkpoints use the
standard ``<experiment>_best.pth`` schema with ``experiment_name``
provenance and are compatible with ``src/evaluate_test.py``.

Usage:
    python -m src.train_nntransunet --config configs/nntransunet_config.yaml --device cuda
    python -m src.train_nntransunet --config configs/nntransunet_config.yaml --smoke-test
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.train import Trainer, get_device, set_seed  # noqa: E402
from src.nntransunet import (  # noqa: E402
    NNTRANSUNET_TRAINING,
    build_nntransunet_training,
    get_nntransunet_train_transforms,
)


def parse_args():
    parser = argparse.ArgumentParser(description="nn-TransUNet training (paper protocol)")
    parser.add_argument("--config", type=str, default="configs/nntransunet_config.yaml",
                        help="Path to YAML config")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name override")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override (paper: 500)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override (paper default: 12)")
    parser.add_argument("--iters-per-epoch", type=int, default=None,
                        help="Iterations-per-epoch override (paper: 250)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override (paper locked: 0.01)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Random init (dry validation only; production runs use pretrained weights)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 1-batch CPU checks (forward/loss/backward + protocol wiring) and exit")
    return parser.parse_args()


def resolve_protocol(train_cfg, args):
    """Resolve the effective protocol; paper values are defaults, CLI overrides are explicit."""
    locked = dict(NNTRANSUNET_TRAINING)
    protocol = {
        "epochs": args.epochs or train_cfg.get("epochs", locked["epochs"]),
        "iters_per_epoch": args.iters_per_epoch or train_cfg.get("iters_per_epoch", locked["iters_per_epoch"]),
        "batch_size": args.batch_size or train_cfg.get("batch_size", locked["batch_size"]),
        "lr": args.lr or train_cfg.get("learning_rate", train_cfg.get("lr", locked["lr"])),
    }
    deviations = []
    for key, locked_key in [("epochs", "epochs"), ("iters_per_epoch", "iters_per_epoch"),
                            ("batch_size", "batch_size"), ("lr", "lr")]:
        if protocol[key] != locked[locked_key]:
            deviations.append(f"{key}={protocol[key]} (paper: {locked[locked_key]})")
    return protocol, deviations


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config.get("model", {}).get("architecture") != "nn_transunet":
        raise ValueError(
            f"Refusing: config architecture is "
            f"{config.get('model', {}).get('architecture')!r}, not 'nn_transunet'. "
            "This runner implements the nn-TransUNet paper protocol only."
        )

    exp_name = args.experiment_name or config.get("experiment_name", "baseline_nntransunet")
    seed = args.seed if args.seed is not None else config.get("random_seed", 42)
    device_pref = args.device or config.get("device", "auto")
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    log_cfg = config.get("logging", {})

    protocol, deviations = resolve_protocol(train_cfg, args)
    set_seed(seed)

    print("=" * 65)
    print("nn-TransUNet paper protocol (Zhao et al., Life 2022;12(10):1570):")
    print(f"  optimizer: SGD + Nesterov momentum 0.99, lr={protocol['lr']}, wd=0.0")
    print(f"  scheduler: PolynomialLR power 0.9, stepped per epoch")
    print(f"  budget: {protocol['epochs']} epochs x {protocol['iters_per_epoch']} iters/epoch "
          f"x batch {protocol['batch_size']}")
    print(f"  model selection: best val_loss; early stopping: DISABLED (paper)")
    if deviations:
        print(f"  DOCUMENTED DEVIATIONS from paper: {'; '.join(deviations)}")
    print("=" * 65)

    if args.no_pretrained:
        config.setdefault("model", {})["pretrained"] = False
        print("NOTE: --no-pretrained given: random init for dry validation only.")

    from src.dataset import ACDCSegDataset, get_val_transforms
    from src.segmentation_model import build_segmentation_model
    from src.losses import DiceCELoss
    from src.encoder import count_parameters

    device = torch.device("cpu") if args.smoke_test else get_device(device_pref)

    processed_dir = data_cfg.get("processed_dir", "data/processed")
    splits_dir = data_cfg.get("splits_dir", "data/splits")
    train_split = data_cfg.get("train_split", str(Path(splits_dir) / "train_patients.txt"))
    val_split = data_cfg.get("val_split", str(Path(splits_dir) / "val_patients.txt"))

    model = build_segmentation_model(config).to(device)
    print(f"Model: {model.__class__.__name__} with {count_parameters(model):,} parameters")

    training = build_nntransunet_training(
        model,
        # Bind CLI-resolved values so pilots/deviations actually take effect:
        # without this, --epochs would shorten training while PolynomialLR
        # still decayed over 500, and --lr would trip the assertion below.
        overrides={"lr": float(protocol["lr"]), "epochs": int(protocol["epochs"])},
    )
    optimizer, scheduler = training["optimizer"], training["scheduler"]
    assert optimizer.__class__.__name__ == "SGD", type(optimizer)
    assert scheduler.__class__.__name__ == "PolynomialLR", type(scheduler)
    assert float(optimizer.param_groups[0]["lr"]) == float(protocol["lr"]), (
        f"Optimizer lr {optimizer.param_groups[0]['lr']} != resolved {protocol['lr']}")
    print(f"Optimizer/scheduler verified: SGD(lr={protocol['lr']}) + PolynomialLR "
          f"(total_iters={training['epochs']})")

    if args.smoke_test:
        print("RUNNING nn-TransUNet CPU SMOKE TEST (1 batch + protocol wiring)")
        dataset = ACDCSegDataset(processed_dir=processed_dir, split_file=train_split)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        logits = model(images)
        assert logits.shape == (2, 4, 256, 256), tuple(logits.shape)
        criterion = DiceCELoss(num_classes=4,
                               dice_weight=loss_cfg.get("dice_weight", 1.0),
                               ce_weight=loss_cfg.get("ce_weight", 1.0),
                               include_background=loss_cfg.get("include_background", False))
        loss = criterion(logits, masks)
        assert torch.isfinite(loss), "Loss is not finite!"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Forward {tuple(logits.shape)} / loss {loss.item():.4f} / backward: SUCCESS")
        print("SMOKE TEST COMPLETE. No full training performed.")
        return

    train_dataset = ACDCSegDataset(
        processed_dir=processed_dir,
        split_file=train_split,
        transform=get_nntransunet_train_transforms(),
    )
    val_dataset = ACDCSegDataset(
        processed_dir=processed_dir,
        split_file=val_split,
        transform=get_val_transforms(),
    )

    # Paper iteration budget without touching the shared Trainer: each epoch
    # draws exactly iters_per_epoch * batch_size samples with replacement.
    n_epoch_samples = protocol["iters_per_epoch"] * protocol["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=protocol["batch_size"],
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=n_epoch_samples),
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )
    assert len(train_loader) == protocol["iters_per_epoch"], (
        f"Loader yields {len(train_loader)} iters/epoch, expected {protocol['iters_per_epoch']}")
    val_loader = DataLoader(
        val_dataset,
        batch_size=protocol["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train iters/epoch verified: {len(train_loader)} "
          f"({n_epoch_samples} samples/epoch from {len(train_dataset)} slices)")

    criterion = DiceCELoss(
        num_classes=data_cfg.get("num_classes", 4),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        include_background=loss_cfg.get("include_background", False),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        mixed_precision=train_cfg.get("mixed_precision", True),
        checkpoint_dir=log_cfg.get("checkpoint_dir", "checkpoints"),
        log_dir=log_cfg.get("log_dir", "results/logs"),
        experiment_name=exp_name,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    from src.train import compute_val_metrics_wrapper
    # patience == n_epochs: early stopping can never trigger (paper: none).
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=protocol["epochs"],
        early_stopping_patience=protocol["epochs"],
        compute_metrics_fn=compute_val_metrics_wrapper,
        monitor_metric="val_loss",
    )


if __name__ == "__main__":
    main()
