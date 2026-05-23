from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset
import torch

import driver


class DriverCliTest(unittest.TestCase):
    def test_parse_args_accepts_four_split_configs(self) -> None:
        args = driver.parse_args(
            [
                "--data-config",
                "config/data/scale_0p0625.yaml",
                "--model-config",
                "config/model/dit.yaml",
                "--framework-config",
                "config/framework/base.yaml",
                "--wrapper-config",
                "config/wrapper/base.yaml",
            ]
        )
        self.assertEqual(args.data_config, "config/data/scale_0p0625.yaml")
        self.assertEqual(args.model_config, "config/model/dit.yaml")
        self.assertEqual(args.framework_config, "config/framework/base.yaml")
        self.assertEqual(args.wrapper_config, "config/wrapper/base.yaml")

    def test_main_rejects_missing_config_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            driver.main(
                [
                    "--data-config",
                    "config/data/scale_0p0625.yaml",
                    "--model-config",
                    "config/model/dit.yaml",
                    "--framework-config",
                    "config/framework/base.yaml",
                    "--wrapper-config",
                    str(Path("config/wrapper/missing.yaml")),
                ]
            )
        self.assertEqual(context.exception.code, 1)

    def test_hot_cache_override_rebuilds_incomplete_loader_with_zero_workers(self) -> None:
        class HotDataset(TensorDataset):
            def requires_single_process_cache_build(self) -> bool:
                return True

        dataset = HotDataset(torch.arange(8))
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )
        runtime = SimpleNamespace(
            objects={"train_dataloader": loader},
            runtime_config={
                "train_dataloader": {
                    "params": {
                        "batch_size": 2,
                        "shuffle": True,
                        "pin_memory": False,
                    }
                }
            },
        )

        changed = driver._apply_hot_cache_dataloader_override(runtime)

        self.assertEqual(changed, ["train_dataloader"])
        rebuilt = runtime.objects["train_dataloader"]
        self.assertEqual(rebuilt.num_workers, 0)
        self.assertIsNone(rebuilt.prefetch_factor)
        self.assertFalse(rebuilt.persistent_workers)
        self.assertIs(rebuilt.dataset, dataset)

    def test_hot_cache_override_leaves_complete_loader_unchanged(self) -> None:
        class CompleteHotDataset(TensorDataset):
            def requires_single_process_cache_build(self) -> bool:
                return False

        dataset = CompleteHotDataset(torch.arange(8))
        loader = DataLoader(dataset, batch_size=2, num_workers=0)
        runtime = SimpleNamespace(
            objects={"train_dataloader": loader},
            runtime_config={"train_dataloader": {"params": {"batch_size": 2}}},
        )

        changed = driver._apply_hot_cache_dataloader_override(runtime)

        self.assertEqual(changed, [])
        self.assertIs(runtime.objects["train_dataloader"], loader)

    def test_hot_cache_override_handles_single_process_loading(self) -> None:
        class CompleteHotDataset(TensorDataset):
            def requires_single_process_cache_build(self) -> bool:
                return False

            def requires_single_process_loading(self) -> bool:
                return True

        dataset = CompleteHotDataset(torch.arange(8))
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )
        runtime = SimpleNamespace(
            objects={"train_dataloader": loader},
            runtime_config={"train_dataloader": {"params": {"batch_size": 2}}},
        )

        changed = driver._apply_hot_cache_dataloader_override(runtime)

        self.assertEqual(changed, ["train_dataloader"])
        self.assertEqual(runtime.objects["train_dataloader"].num_workers, 0)
        self.assertFalse(runtime.objects["train_dataloader"].persistent_workers)


if __name__ == "__main__":
    unittest.main()
