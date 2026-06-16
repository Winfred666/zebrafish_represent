"""Rewrite the upstream TRELLIS sparse-structure flow checkpoint for the local full-canvas model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules.model.trellis_ss_flow import (
    TRELLISSparseStructureFlow,
    _build_input_layer_weight,
    _build_pos_emb,
    _build_zero_linear_weight,
)
from utils.runtime_factory import load_yaml_config
from utils.sanitize.model_config import TRELLISSparseStructureFlowParams

DEFAULT_SOURCE_PATH = (
    "result/checkpoints/sparse_structure/ss_flow_img_dit_L_16l8_fp16.safetensors"
)
DEFAULT_OUTPUT_PATH = (
    "result/checkpoints/sparse_structure/"
    "ss_flow_img_dit_L_fullcanvas_patch16_localwarm.safetensors"
)
GEOMETRY_REWRITE_KEYS = {
    "pos_emb",
    "input_layer.weight",
    "input_layer.bias",
    "out_layer.weight",
    "out_layer.bias",
}


def _load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("rewrite_trellis_ss_flow_ckpt.py requires safetensors.") from exc
    return load_file(str(path), device="cpu")


def _save_safetensors(state_dict: dict[str, torch.Tensor], path: str | Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError("rewrite_trellis_ss_flow_ckpt.py requires safetensors.") from exc
    save_file(state_dict, str(path))


def _load_model_section(model_config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(str(model_config_path))
    model_section = config["model"]
    if model_section["class_name"] != "TRELLISSparseStructureFlow":
        raise ValueError(
            "rewrite_trellis_ss_flow_ckpt.py expects a TRELLISSparseStructureFlow model section. "
            f"Got class_name={model_section['class_name']}"
        )
    return model_section


def _build_meta_model(params: TRELLISSparseStructureFlowParams) -> TRELLISSparseStructureFlow:
    kwargs = params.model_dump(mode="python")
    kwargs["load_from_ckpt"] = None
    kwargs["strict_load"] = False
    with torch.device("meta"):
        return TRELLISSparseStructureFlow(**kwargs)


def _build_geometry_tensor(
    key: str,
    params: TRELLISSparseStructureFlowParams,
    target_shape: torch.Size,
    dtype: torch.dtype,
) -> torch.Tensor:
    if key == "pos_emb":
        return _build_pos_emb(
            math.prod(size // params.patch_size for size in params.input_size),
            params.hidden_size,
            dtype=dtype,
        )
    if key == "input_layer.weight":
        return _build_input_layer_weight(
            params.hidden_size,
            params.patch_size ** 3 * params.in_channels,
            dtype=dtype,
        )
    if key == "input_layer.bias":
        return torch.zeros(target_shape, dtype=dtype)
    if key == "out_layer.weight":
        return _build_zero_linear_weight(
            params.patch_size ** 3 * params.out_channels,
            params.hidden_size,
            dtype=dtype,
        )
    if key == "out_layer.bias":
        return torch.zeros(target_shape, dtype=dtype)
    raise KeyError(f"Unexpected geometry rewrite key: {key}")


def rewrite_checkpoint(
    *,
    source_path: str | Path,
    output_path: str | Path,
    model_section: dict[str, Any],
) -> dict[str, Any]:
    params = TRELLISSparseStructureFlowParams.model_validate(model_section["params"])
    source_state = _load_safetensors(source_path)
    meta_model = _build_meta_model(params)
    target_state = meta_model.state_dict()

    rewritten_state: dict[str, torch.Tensor] = {}
    kept_keys: list[str] = []
    rewritten_keys: list[str] = []
    missing_source_keys: list[str] = []

    for key, target_value in target_state.items():
        target_shape = target_value.shape
        source_value = source_state.get(key)
        if key in GEOMETRY_REWRITE_KEYS:
            rewritten_state[key] = _build_geometry_tensor(
                key,
                params,
                target_shape,
                source_value.dtype if source_value is not None else torch.float32,
            )
            rewritten_keys.append(key)
            continue
        if source_value is not None and source_value.shape == target_shape:
            rewritten_state[key] = source_value.detach().cpu()
            kept_keys.append(key)
            continue
        if source_value is None:
            missing_source_keys.append(key)
            continue
        raise ValueError(
            "Found unexpected incompatible checkpoint tensor outside the geometry-dependent keys. "
            f"Key={key}, source_shape={tuple(source_value.shape)}, target_shape={tuple(target_shape)}"
        )

    if missing_source_keys:
        raise ValueError(
            "Source checkpoint is missing non-geometry tensors required by the local model: "
            f"{missing_source_keys[:8]}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_safetensors(rewritten_state, output_path)

    return {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "kept_keys": kept_keys,
        "dropped_keys": rewritten_keys,
        "num_source_keys": len(source_state),
        "num_output_keys": len(rewritten_state),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="config/model/trellis_ss_flow.yaml")
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_section = _load_model_section(args.model_config)
    summary = rewrite_checkpoint(
        source_path=args.source_path,
        output_path=args.output_path,
        model_section=model_section,
    )
    print(
        "Rewrote TRELLIS sparse-structure flow checkpoint to "
        f"{summary['output_path']} (kept={len(summary['kept_keys'])}, "
        f"rewritten={len(summary['dropped_keys'])}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
