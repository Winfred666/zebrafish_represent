# zebrafish_represent
Pretraining generalist base model for Zebrafish 3D volume neural representation.

This project implements a self-supervised learning pipeline for 3D zebrafish volumes using masked volume reconstruction with a 3D U-Net architecture.

## Features

- **TIF to Volume Conversion**: Convert and downsample .tif files to .npy volumes
- **Pretext Dataset Generation**: Generate random masked train/val datasets for self-supervised learning
- **3D U-Net Architecture**: Standard 3D U-Net implementation for volumetric data
- **PyTorch Lightning Training**: Scalable training with Lightning framework
- **Configurable Pipeline**: YAML-based configuration for easy experimentation

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
zebrafish_represent/
├── config/
│   └── default.yaml          # Example configuration file
├── data/                      # Data directory (not tracked)
├── model/
│   ├── unet3d.py             # 3D U-Net implementation
│   └── lightning/
│       └── fish_module.py    # Lightning training module
├── utils/
│   ├── tif2volume.py         # TIF to volume conversion
│   ├── gen_pretext_dataset.py # Dataset generation
│   └── load_config.py        # Config loading utility
├── driver.py                  # Main training script
└── requirements.txt           # Python dependencies
```

## Usage

### 1. Convert TIF Files to Volumes

Convert .tif files to downsampled .npy volumes:

```bash
# Single file
python utils/tif2volume.py --input data/sample.tif --output data/sample.npy --scale 0.5 0.5 0.5

# Batch process directory
python utils/tif2volume.py --input data/raw/ --output data/volumes/ --batch --scale 0.5 0.5 0.5
```

Options:
- `--scale`: Downsampling scale factors for (depth, height, width)
- `--no-normalize`: Disable intensity normalization
- `--batch`: Process all .tif files in directory

### 2. Generate Pretext Datasets

Generate random masked train/val datasets:

```bash
python utils/gen_pretext_dataset.py \
    --volume-dir data/volumes \
    --output-dir data \
    --train-ratio 0.8 \
    --samples-per-volume 10 \
    --mask-type block \
    --mask-ratio 0.3
```

Options:
- `--volume-dir`: Directory containing .npy volume files
- `--output-dir`: Directory to save train.pt and val.pt
- `--train-ratio`: Ratio of data for training (default: 0.8)
- `--samples-per-volume`: Number of masked samples per volume (default: 10)
- `--mask-type`: Masking strategy ("block" or "patch")
- `--mask-ratio`: Ratio of volume to mask (default: 0.3)
- `--block-size-min/max`: Block size range for block masking
- `--patch-size`: Patch size for patch masking

### 3. Train the Model

Train the 3D U-Net using a configuration file:

```bash
python driver.py --config config/default.yaml
```

The configuration file controls all aspects of training including:
- Model architecture (channels, features, etc.)
- Training hyperparameters (learning rate, epochs, etc.)
- Data loading (batch size, paths, etc.)
- Trainer settings (accelerator, precision, etc.)

### 4. Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

## Configuration

Edit `config/default.yaml` to customize training:

```yaml
data:
  train_path: "data/train.pt"
  val_path: "data/val.pt"
  batch_size: 4

model:
  in_channels: 1
  out_channels: 1
  features: [64, 128, 256, 512]

training:
  learning_rate: 0.0001
  max_epochs: 100
```

## Model Architecture

The 3D U-Net consists of:
- **Encoder**: Downsampling path with max pooling
- **Decoder**: Upsampling path with skip connections
- **Features**: Configurable feature channels at each level (default: [64, 128, 256, 512])

The model learns to reconstruct masked regions of 3D volumes through self-supervised learning.

## Loss Function

The training uses a composite loss:
- Reconstruction loss on entire volume
- Weighted masked region loss (2x weight)
- Unmasked region loss

This encourages the model to focus on reconstructing masked areas while preserving visible regions.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- scikit-image
- numpy
- pyyaml
- tifffile

## License

See LICENSE file for details.
