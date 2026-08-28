"""Display and artifact/logging helper package.

Keep this package init lightweight to avoid importing optional visualization
dependencies during training-only code paths.
"""

from utils.display.log_artifact import ArtifactManager, log_image_artifact
from utils.display.log_gpu import IntegratedGPUMemoryMonitor
from utils.display.transformer_diagnostics import (
    capture_transformer_attention,
    log_transformer_diagnostics,
    should_log_gradient_histograms,
)
from utils.display.visualize_2d import (
    build_clipped_midw_grid,
    build_w_mip_grid,
    fix_2d_scalar,
    render_slice,
)

__all__ = [
    "ArtifactManager",
    "IntegratedGPUMemoryMonitor",
    "build_clipped_midw_grid",
    "build_w_mip_grid",
    "capture_transformer_attention",
    "fix_2d_scalar",
    "log_transformer_diagnostics",
    "render_slice",
    "log_image_artifact",
    "should_log_gradient_histograms",
]
