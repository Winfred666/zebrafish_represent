from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import driver


class DriverRunModeTest(unittest.TestCase):
    def _runtime(self, *, run_mode: str) -> SimpleNamespace:
        trainer = mock.Mock()
        artifact_manager = mock.Mock()
        return SimpleNamespace(
            runtime_config={
                "run_mode": run_mode,
                "resume_ckpt_path": "/tmp/example.ckpt",
            },
            logger=None,
            objects={"framework": object()},
            train_loader="train_loader",
            val_loader="val_loader",
            trainer=trainer,
            artifact_manager=artifact_manager,
            paths=SimpleNamespace(
                data="data_cfg",
                model="model_cfg",
                framework="framework_cfg",
                wrapper="wrapper_cfg",
            ),
        )

    @mock.patch("driver._log_config_params")
    @mock.patch("driver.build_training_runtime_from_files")
    @mock.patch("driver.load_dotenv")
    def test_validate_run_mode_calls_trainer_validate(
        self,
        mock_load_dotenv,
        mock_build_runtime,
        mock_log_config_params,
    ) -> None:
        runtime = self._runtime(run_mode="validate")
        mock_build_runtime.return_value = runtime

        driver.train(
            data_config_path="data.yaml",
            model_config_path="model.yaml",
            framework_config_path="framework.yaml",
            wrapper_config_path="wrapper.yaml",
        )

        runtime.trainer.validate.assert_called_once_with(
            runtime.objects["framework"],
            dataloaders=runtime.val_loader,
            ckpt_path="/tmp/example.ckpt",
        )
        runtime.trainer.fit.assert_not_called()
        runtime.artifact_manager.cleanup_temp_folder.assert_called_once_with()
        mock_load_dotenv.assert_called_once_with(".env")
        mock_log_config_params.assert_not_called()

    @mock.patch("driver._log_config_params")
    @mock.patch("driver.build_training_runtime_from_files")
    @mock.patch("driver.load_dotenv")
    def test_default_fit_run_mode_calls_trainer_fit(
        self,
        mock_load_dotenv,
        mock_build_runtime,
        mock_log_config_params,
    ) -> None:
        runtime = self._runtime(run_mode="fit")
        runtime.logger = object()
        mock_build_runtime.return_value = runtime

        driver.train(
            data_config_path="data.yaml",
            model_config_path="model.yaml",
            framework_config_path="framework.yaml",
            wrapper_config_path="wrapper.yaml",
        )

        runtime.trainer.fit.assert_called_once_with(
            runtime.objects["framework"],
            train_dataloaders=runtime.train_loader,
            val_dataloaders=runtime.val_loader,
            ckpt_path="/tmp/example.ckpt",
        )
        runtime.trainer.validate.assert_not_called()
        runtime.artifact_manager.cleanup_temp_folder.assert_called_once_with()
        mock_load_dotenv.assert_called_once_with(".env")
        mock_log_config_params.assert_called_once_with(runtime.logger, runtime.runtime_config)


if __name__ == "__main__":
    unittest.main()
