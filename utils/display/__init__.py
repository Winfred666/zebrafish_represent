"""Display and artifact/logging helper package.

Keep this package init lightweight to avoid importing optional visualization
dependencies during training-only code paths.
"""

from utils.display.log_artifact import ArtifactManager, log_image_artifact
from utils.display.log_gpu import IntegratedGPUMemoryMonitor
from utils.display.visualize_2d import fix_2d_scalar, render_slice

__all__ = [
    "ArtifactManager",
    "IntegratedGPUMemoryMonitor",
    "fix_2d_scalar",
    "render_slice",
    "log_image_artifact",
]
