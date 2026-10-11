"""Regression tests for the stdlib ``ssl`` module-name collision.

Root cause (nnU-Net baseline audit): the repository previously contained
``src/ssl.py``. Any process with ``src/`` at the front of ``sys.path``
(e.g. ``python src/nnunet_baseline.py``) resolved top-level ``import ssl``
to the project file instead of the standard library, breaking ``urllib``,
PyTorch (via ``asyncio.sslproto``), and other stdlib consumers with
``AttributeError: partially initialized module 'ssl' ...``.

The project module was renamed to ``src/ssl_pretraining.py``. These tests
prove (1) stdlib ``ssl`` always resolves correctly, (2) the reported
``nnunet_baseline.py --help`` failure mode stays fixed, and (3) the project
SSL functionality remains importable under its new name.

Lightweight: subprocess checks import stdlib/nibabel only; no training, no
downloads, no checkpoints, no test data.
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestStdlibShadowing(unittest.TestCase):
    def test_no_src_module_shadows_stdlib(self):
        """No file directly under src/ may share a stdlib top-level name."""
        offenders = sorted(
            f.stem for f in (REPO_ROOT / "src").glob("*.py")
            if f.stem in sys.stdlib_module_names
        )
        self.assertEqual(offenders, [], f"shadowing modules: {offenders}")

    def test_stdlib_ssl_with_src_on_path(self):
        """With src/ first on sys.path, `import ssl` must be the stdlib."""
        code = (
            "import sys; sys.path.insert(0, 'src');"
            "import ssl;"
            "assert hasattr(ssl, 'SSLWantReadError'), ssl.__file__;"
            "assert 'motion_guided_ssl_project' not in ssl.__file__.replace(chr(92), '/'), ssl.__file__;"
            "print('STDLIB-SSL-OK:', ssl.__file__)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("STDLIB-SSL-OK", result.stdout)

    def test_nnunet_baseline_help_from_script_dir(self):
        """Reported failure mode: `python src/nnunet_baseline.py --help`."""
        result = subprocess.run(
            [sys.executable, "src/nnunet_baseline.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("export", result.stdout)

    def test_project_ssl_module_importable(self):
        """Project SSL API intact under the new module name; stdlib unaffected."""
        from src.ssl_pretraining import (
            SSLModel,
            SSLTrainer,
            build_ssl_model,
            compute_ssl_loss,
            create_ssl_pipeline,
        )
        for obj in (SSLModel, SSLTrainer, build_ssl_model,
                    compute_ssl_loss, create_ssl_pipeline):
            self.assertTrue(callable(obj), obj)
        import ssl
        self.assertTrue(hasattr(ssl, "SSLContext"))


if __name__ == "__main__":
    unittest.main()
