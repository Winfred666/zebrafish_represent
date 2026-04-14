"""Display and artifact/logging helper package.

Keep this package init lightweight to avoid importing optional visualization
dependencies during training-only code paths.
"""

from utils.display.log_artifact import ArtifactManager, log_image_artifact, prepare_train_artifacts
from utils.display.log_gpu import IntegratedGPUMemoryMonitor, build_gpu_memory_callback

__all__ = [
    "ArtifactManager",
    "IntegratedGPUMemoryMonitor",
    "build_gpu_memory_callback",
    "prepare_train_artifacts",
    "log_image_artifact",
]
