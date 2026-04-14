from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import driver


class DriverBootstrapTest(unittest.TestCase):
    def test_auto_probe_failure_masks_cuda(self) -> None:
        config_path = Path("config/data/scale_0p0625.yaml")
        with mock.patch("driver._probe_cuda_runtime", return_value=False):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
                accelerator = driver._bootstrap_runtime_environment(config_path)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")

        self.assertEqual(accelerator, "auto")

    def test_explicit_gpu_requires_usable_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "gpu.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    trainer:
                      accelerator: gpu
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch("driver._probe_cuda_runtime", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "requires a usable CUDA runtime"):
                    driver._bootstrap_runtime_environment(config_path)


if __name__ == "__main__":
    unittest.main()
