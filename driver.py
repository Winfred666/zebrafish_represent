"""
Main driver script for training the 3D U-Net model on zebrafish volumes.
"""
import argparse
import sys
from pathlib import Path

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.load_config import load_config
from model.lightning.fish_module import FishModule, create_dataloaders


def train(config_path: str):
    """
    Train the model using configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded:")
    print(config)
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    L.seed_everything(seed)
    
    # Data configuration
    data_config = config.get('data', {})
    train_path = data_config.get('train_path', 'data/train.pt')
    val_path = data_config.get('val_path', 'data/val.pt')
    batch_size = data_config.get('batch_size', 4)
    num_workers = data_config.get('num_workers', 4)
    
    # Model configuration
    model_config = config.get('model', {})
    in_channels = model_config.get('in_channels', 1)
    out_channels = model_config.get('out_channels', 1)
    features = model_config.get('features', [64, 128, 256, 512])
    trilinear = model_config.get('trilinear', False)
    use_batchnorm = model_config.get('use_batchnorm', True)
    loss_type = model_config.get('loss_type', 'mse')
    
    # Training configuration
    train_config = config.get('training', {})
    learning_rate = train_config.get('learning_rate', 1e-4)
    weight_decay = train_config.get('weight_decay', 1e-5)
    max_epochs = train_config.get('max_epochs', 100)
    
    # Checkpoint configuration
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_dir = checkpoint_config.get('dir', 'checkpoints')
    save_top_k = checkpoint_config.get('save_top_k', 3)
    
    # Logging configuration
    log_config = config.get('logging', {})
    log_dir = log_config.get('dir', 'logs')
    experiment_name = log_config.get('experiment_name', 'zebrafish_unet')
    
    print(f"\nLoading data from:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        train_path=train_path,
        val_path=val_path if Path(val_path).exists() else None,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('val', None)
    
    print(f"\nDataset info:")
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = FishModule(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        trilinear=trilinear,
        use_batchnorm=use_batchnorm,
        loss_type=loss_type
    )
    
    print(f"Model parameters: {model.model.get_num_params():,}")
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{experiment_name}-{{epoch:02d}}-{{val/loss:.4f}}',
        monitor='val/loss' if val_loader else 'train/loss',
        mode='min',
        save_top_k=save_top_k,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Early stopping (if validation set exists)
    if val_loader and train_config.get('early_stopping', True):
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            patience=train_config.get('patience', 10),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name
    )
    
    # Setup trainer
    trainer_config = config.get('trainer', {})
    accelerator = trainer_config.get('accelerator', 'auto')
    devices = trainer_config.get('devices', 'auto')
    precision = trainer_config.get('precision', '32')
    gradient_clip_val = trainer_config.get('gradient_clip_val', 0.5)
    accumulate_grad_batches = trainer_config.get('accumulate_grad_batches', 1)
    
    print(f"\nTraining configuration:")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Accelerator: {accelerator}")
    print(f"  Devices: {devices}")
    print(f"  Precision: {precision}")
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        deterministic=train_config.get('deterministic', False)
    )
    
    # Train the model
    print("\nStarting training...\n")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    print(f"\nTraining complete!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/loss: {checkpoint_callback.best_model_score:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net on zebrafish volumes"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Start training
    train(args.config)


if __name__ == "__main__":
    main()
