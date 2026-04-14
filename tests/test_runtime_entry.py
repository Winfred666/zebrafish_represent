from __future__ import annotations

import unittest

import torch

from modules.rect_flow import RectifiedFlowModule, create_rectified_flow_dataloaders
from utils.sanitize.load_config import load_config
from utils.sanitize.runtime_config import RectifiedFlowComputeConfig


class RuntimeEntryTest(unittest.TestCase):
    def test_scale_entry_loads_and_builds_volume_module(self) -> None:
        config = load_config('config/data/scale_0p0625.yaml')
        self.assertEqual(config.data.scale_factor, (0.0625, 0.0625, 0.0625))
        self.assertEqual(config.model.input_size, (4, 4, 4))
        self.assertEqual(config.train.framework, 'rectified_flow')
        self.assertIsInstance(config.framework_config, RectifiedFlowComputeConfig)
        self.assertEqual(config.framework_config.train_loader.batch_size, config.data.batch_size)
        self.assertEqual(config.framework_config.train_loader.dataset.data_dir, config.data.train_dir)
        self.assertEqual(
            config.framework_config.train_loader.dataset.samples_per_volume,
            config.data.samples_per_volume_train,
        )
        self.assertEqual(config.framework_config.model.in_channels, config.model.in_channels)

        dataloaders = create_rectified_flow_dataloaders(config.framework_config)
        batch = next(iter(dataloaders['train']))
        self.assertEqual(tuple(batch['target'].shape), (1, 1, 4, 4, 4))

        module = RectifiedFlowModule(config.framework_config)
        timesteps = torch.rand(1)
        output = module(batch['target'], timesteps)
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4, 4))


if __name__ == '__main__':
    unittest.main()
