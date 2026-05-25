from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
