# Copilot instructions for `zebrafish_represent`

## Big picture (data + training flow)

- This repo trains a **self-supervised 3D Zebrafish Generalist** to represent any 3D zebrafish volumes. Main way to pretrain this general encoder is to use pretext task like reconstruct masked zebrafish volumes.

- Pipeline is:
  1) Convert raw microscopy `.tif/.tiff` → downsampled normalized `.npy` volumes via `utils/tif2volume.py`.
  2) Generate masked pretext samples and save as `train.pt` / `val.pt` via `utils/gen_pretext_dataset.py`.
     - Each sample is a dict with **`input`**, **`target`**, **`mask`** tensors.
     - Shapes are channel-first: `input/target: (C,D,H,W)`, `mask: (1,D,H,W)`.
  3) Train with PyTorch Lightning entrypoint `driver.py` using YAML config in `config/default.yaml`.

## Key modules and conventions

- Model: Any advanced 3D vision backbone models are welcomed, but input/output must be consistent. For example, `model/unet3d.py` defines `UNet3D` (expects `(B,C,D,H,W)` tensors).

- Lightning: One lightning module represents one training task. It can accept any backbone models defined above. For example, `model/lightning/fish_module.py` applied a basic masked reconstruction task.

- Dataset and Results: DO NOT open nor read contents of `data/**/*` or `results/**/*` as they are huge or complicated. Only scan the folder structure and file name when we want to use specific files to run script or set config.

- Virtual python env: DO NOT visit .venv as env is huge.

## How to run (what’s “real” in this repo)

The project use uv for env and dependency management; If running in commandline, do not use `<< PY` command, use `-c "..."` instead. 

- Training is config-driven, `uv run driver.py --config config/default.yaml`.

- Dataset generation: `uv run utils/gen_pretext_dataset.py --volume-dir data/volumes --output-dir data ...`.

- TIF conversion: `uv run utils/tif2volume.py --input <file-or-dir> --output <file-or-dir> --scale 0.5 0.5 0.5 [--batch]`.

## Patterns to follow when editing

- Keep tensors channel-first and 5D in the network: `(B,C,D,H,W)`; dataset items stay 4D per-sample.

- When manipulate index and coordinates, REMEMBER (D,H,W) == (x, y, z), D->x , H->y , W->z, this always make x in first dim.

- If you change the sample dict schema, update both `utils/gen_pretext_dataset.py` and `PretextDataset/FishModule.compute_loss()`.

- Prefer extending behavior via YAML config keys read in `driver.py` and `config/**/*.yaml` (e.g., adding a new loss type).