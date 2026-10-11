"""Focused tests for baseline evaluation dispatch (synthetic only).

Covers: run-ID validation, checkpoint resolution layouts, fraction/seed
resolution, builder kwarg validation, provenance gating, and an end-to-end
synthetic evaluate_run + qualitative loader pass on temp data.

No training, no real test data, no downloads, no expensive models.

Run:  python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch

from src import evaluate_test as ev
from src import generate_qualitative_results as qr
from src.baseline_models import build_baseline_model


class TestRunIdValidation(unittest.TestCase):
    def test_parse_run_id_legacy_unchanged(self):
        self.assertEqual(ev.parse_run_id("full_pipeline_10pct_seed42"), (10, 42))
        for bad in [
            "nope",
            "full_pipeline_10pct_seed42_smoke",
            "unet_100pct_seed42",
        ]:
            with self.assertRaises(ValueError, msg=bad):
                ev.parse_run_id(bad)

    def test_safe_run_id_pattern(self):
        for good in ["unet_100pct_seed42", "my-run.v2", "a"]:
            self.assertEqual(ev.validate_safe_run_id(good), good)
        for bad in ["", "../evil", "/abs/path", "a b", "a/b", ".hidden", "-x"]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                ev.validate_safe_run_id(bad)

    def test_resolve_checkpoint_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                ev.resolve_checkpoint(
                    "full_pipeline_10pct_seed42",
                    checkpoint_dir=str(tmp / "none"),
                )
            with self.assertRaises(FileNotFoundError):
                ev.resolve_checkpoint(
                    "custom_run",
                    checkpoint_path_override=str(tmp / "nope_best.pth"),
                )
            bad = tmp / "run_latest.pth"
            bad.write_text("x")
            with self.assertRaises(ValueError):
                ev.resolve_checkpoint(
                    "custom_run", checkpoint_path_override=str(bad)
                )
            good = tmp / "run_best.pth"
            good.write_text("x")
            self.assertEqual(
                ev.resolve_checkpoint(
                    "custom_run", checkpoint_path_override=str(good)
                ),
                good,
            )
            d = tmp / "ckpts" / "full_pipeline_10pct_seed42"
            d.mkdir(parents=True)
            legacy = d / "full_pipeline_10pct_seed42_best.pth"
            legacy.write_text("x")
            self.assertEqual(
                ev.resolve_checkpoint(
                    "full_pipeline_10pct_seed42",
                    checkpoint_dir=str(tmp / "ckpts"),
                ),
                legacy,
            )

    def test_resolve_fraction_seed(self):
        self.assertEqual(
            ev.resolve_fraction_seed("full_pipeline_25pct_seed42", None, None),
            (25, 42),
        )
        self.assertEqual(
            ev.resolve_fraction_seed("custom_run", 100, 7), (100, 7)
        )
        with self.assertRaises(ValueError):
            ev.resolve_fraction_seed("custom_run", None, 1)
        with self.assertRaises(ValueError):
            ev.resolve_fraction_seed("custom_run", 100, None)
        with self.assertRaises(ValueError):  # unknown fraction
            ev.resolve_fraction_seed("custom_run", 95, 1)


class TestBuilderKwargs(unittest.TestCase):
    def test_unexpected_kwarg_rejected(self):
        with self.assertRaises(TypeError) as ctx:
            build_baseline_model("unetpp", prior_boxes={})
        self.assertIn("prior_boxes", str(ctx.exception))

    def test_kwargs_builder_accepts_no_kwargs(self):
        # Attention U-Net builder takes **kwargs; cheap CPU model, no download.
        model = build_baseline_model("attention_unet")
        self.assertEqual(type(model).__name__, "AttentionUNet")

    def test_unknown_architecture(self):
        with self.assertRaises(ValueError):
            build_baseline_model("nope_arch")


class TestSyntheticEndToEnd(unittest.TestCase):
    """Synthetic evaluate_run + qualitative loader on temp data only."""

    def _make_env(self, tmp):
        tmp = Path(tmp)
        proc = tmp / "proc"
        proc.mkdir()
        split = tmp / "split.txt"
        split.write_text("patient901\npatient902\n")
        rng = np.random.RandomState(0)
        for pid in ["patient901", "patient902"]:
            for s in range(2):
                mask = np.zeros((32, 32), dtype=np.int64)
                mask[8:20, 8:20] = 1
                mask[4:8, 4:8] = 2
                np.savez(
                    proc / f"{pid}_frame00_slice0{s}.npz",
                    image=rng.randn(32, 32).astype(np.float32),
                    mask=mask,
                    patient_id=pid,
                    slice_idx=s,
                    frame_idx=0,
                )
        from src.segmentation_model import SegmentationUNet

        model = SegmentationUNet(in_channels=1, num_classes=4)
        run_id = "patient901_unet_100"
        ckpt = tmp / f"{run_id}_best.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "experiment_name": run_id,
            },
            ckpt,
        )
        config = {
            "data": {
                "processed_dir": str(proc),
                "test_split": str(split),
                "num_workers": 0,
                "num_classes": 4,
            },
            "model": {
                "architecture": "2d_unet",
                "in_channels": 1,
                "num_classes": 4,
            },
            "logging": {
                "checkpoint_dir": str(tmp / "ckpts"),
                "results_dir": str(tmp / "results"),
            },
        }
        return tmp, proc, split, run_id, ckpt, config, model

    def test_evaluate_run_and_provenance(self):
        prev = ev.CANONICAL_TEST_SPLIT
        with tempfile.TemporaryDirectory() as tmp:
            tmp, proc, split, run_id, ckpt, config, model = self._make_env(tmp)
            ev.CANONICAL_TEST_SPLIT = split
            try:
                device = torch.device("cpu")
                result = ev.evaluate_run(
                    run_id,
                    config,
                    device,
                    batch_size=2,
                    processed_dir_override=str(proc),
                    checkpoint_path_override=str(ckpt),
                    label_fraction_override=100,
                    seed_override=42,
                )
                self.assertEqual(result["run_id"], run_id)
                self.assertEqual(result["label_fraction"], 100)
                self.assertEqual(result["seed"], 42)
                self.assertEqual(result["n_patients"], 2)
                self.assertEqual(result["n_slices"], 4)
                self.assertTrue(
                    (
                        Path(tmp)
                        / "results"
                        / run_id
                        / "test_metrics.json"
                    ).exists()
                )
                # Provenance mismatch refused.
                bad = Path(tmp) / "other_best.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "experiment_name": "someone_else",
                    },
                    bad,
                )
                with self.assertRaises(ValueError):
                    ev.evaluate_run(
                        run_id,
                        config,
                        device,
                        batch_size=2,
                        processed_dir_override=str(proc),
                        checkpoint_path_override=str(bad),
                        label_fraction_override=100,
                        seed_override=42,
                    )
                # Missing provenance refused (fail-closed).
                noprovenance = Path(tmp) / "noprovenance_best.pth"
                torch.save({"model_state_dict": model.state_dict()},
                           noprovenance)
                with self.assertRaises(ValueError):
                    ev.evaluate_run(
                        run_id,
                        config,
                        device,
                        batch_size=2,
                        processed_dir_override=str(proc),
                        checkpoint_path_override=str(noprovenance),
                        label_fraction_override=100,
                        seed_override=42,
                    )
                # Legacy production path still gated on checkpoint presence.
                with self.assertRaises(FileNotFoundError):
                    ev.evaluate_run("full_pipeline_10pct_seed42", config,
                                    device)
            finally:
                ev.CANONICAL_TEST_SPLIT = prev

    def test_qualitative_loader(self):
        prev = ev.CANONICAL_TEST_SPLIT
        with tempfile.TemporaryDirectory() as tmp:
            tmp, proc, split, run_id, ckpt, config, model = self._make_env(tmp)
            ev.CANONICAL_TEST_SPLIT = split
            try:
                device = torch.device("cpu")
                loaded = qr.load_production_model(
                    run_id,
                    config,
                    device,
                    checkpoint_dir=str(Path(tmp) / "ckpts2"),
                    checkpoint_path_override=str(ckpt),
                )
                self.assertEqual(type(loaded).__name__, "SegmentationUNet")
                bad = Path(tmp) / "other_best.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "experiment_name": "someone_else",
                    },
                    bad,
                )
                with self.assertRaises(ValueError):
                    qr.load_production_model(
                        run_id,
                        config,
                        device,
                        checkpoint_dir=str(Path(tmp) / "ckpts2"),
                        checkpoint_path_override=str(bad),
                    )
                # Missing provenance refused on the override path,
                # consistent with src/evaluate_test.py (fail-closed).
                noprovenance = Path(tmp) / "noprovenance_best.pth"
                torch.save({"model_state_dict": model.state_dict()},
                           noprovenance)
                with self.assertRaises(ValueError):
                    qr.load_production_model(
                        run_id,
                        config,
                        device,
                        checkpoint_dir=str(Path(tmp) / "ckpts2"),
                        checkpoint_path_override=str(noprovenance),
                    )
                # Missing provenance refused on the legacy default-layout
                # path as well: no checkpoint the evaluator would reject
                # may be visualized.
                legacy_dir = Path(tmp) / "legacy_ckpts" / run_id
                legacy_dir.mkdir(parents=True)
                legacy_ckpt = legacy_dir / f"{run_id}_best.pth"
                torch.save({"model_state_dict": model.state_dict()},
                           legacy_ckpt)
                with self.assertRaises(ValueError):
                    qr.load_production_model(
                        run_id,
                        config,
                        device,
                        checkpoint_dir=str(Path(tmp) / "legacy_ckpts"),
                    )
            finally:
                ev.CANONICAL_TEST_SPLIT = prev


if __name__ == "__main__":
    unittest.main()
