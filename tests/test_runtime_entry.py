from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from modules.rect_flow import RectifiedFlowModule, create_rectified_flow_dataloaders
from utils.sanitize.param_class import RectifiedFlowParams
from utils.sanitize.runtime_factory import build_framework_runtime, load_split_configs, sanitize_split_configs


class RuntimeEntryTest(unittest.TestCase):
    def test_scale_entry_loads_and_builds_volume_module(self) -> None:
        _, data_config, model_config, framework_config, wrapper_config = load_split_configs(
            data_config_path="config/data/scale_0p0625.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        configs = sanitize_split_configs(
            data_config=data_config,
            model_config=model_config,
            framework_config=framework_config,
            wrapper_config=wrapper_config,
        )
        self.assertEqual(configs.data.scale_factor, (0.0625, 0.0625, 0.0625))
        self.assertEqual(configs.model.input_size, (4, 4, 4))
        self.assertEqual(configs.framework.framework, "rectified_flow")

        built_framework = build_framework_runtime(configs)
        self.assertIsInstance(built_framework.params, RectifiedFlowParams)
        self.assertEqual(built_framework.params.train_loader.batch_size, configs.data.batch_size)
        self.assertEqual(built_framework.params.train_loader.dataset.data_dir, configs.data.train_dir)
        self.assertEqual(
            built_framework.params.train_loader.dataset.samples_per_volume,
            configs.data.samples_per_volume_train,
        )
        self.assertEqual(built_framework.params.model.in_channels, configs.model.in_channels)

        dataloaders = create_rectified_flow_dataloaders(built_framework.params)
        batch = next(iter(dataloaders['train']))
        self.assertEqual(tuple(batch['target'].shape), (1, 1, 4, 4, 4))

        module = RectifiedFlowModule(built_framework.params)
        timesteps = torch.rand(1)
        output = module(batch['target'], timesteps)
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4, 4))

    def test_wrapper_rejects_missing_resume_checkpoint(self) -> None:
        _, data_config, model_config, framework_config, wrapper_config = load_split_configs(
            data_config_path="config/data/scale_0p0625.yaml",
            model_config_path="config/model/base.yaml",
            framework_config_path="config/framework/base.yaml",
            wrapper_config_path="config/wrapper/base.yaml",
        )
        wrapper_with_missing_resume = dict(wrapper_config)
        wrapper_with_missing_resume["resume_ckpt_path"] = str(Path(tempfile.gettempdir()) / "missing-resume.ckpt")
        with self.assertRaises(FileNotFoundError):
            sanitize_split_configs(
                data_config=data_config,
                model_config=model_config,
                framework_config=framework_config,
                wrapper_config=wrapper_with_missing_resume,
            )


if __name__ == '__main__':
    unittest.main()
