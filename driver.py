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
    # Load config (load_config returns a normalized dict with defaults applied)
    config = load_config(config_path)
    print("Configuration loaded:")
    print(config)

    # Set random seed for reproducibility
    L.seed_everything(config['seed'])

    data_config = config['data']
    model_config = config['model']
    train_config = config['training']
    trainer_config = config['trainer']
    checkpoint_config = config['checkpoint']
    log_config = config['logging']

    print(f"\nLoading data from:")
    print(f"  Train: {data_config['train_path']}")
    print(f"  Val: {data_config['val_path']}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(**data_config)
    
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('val', None)
    
    print(f"\nDataset info:")
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")
    
    # Initialize model
    print(f"\nInitializing model...")
    # FishModule expects a flat kwargs dict across model + training.
    model = FishModule(**{**model_config, **train_config})
    
    print(f"Model parameters: {model.model.get_num_params():,}")
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_config['dir'],
        filename=f"{log_config['experiment_name']}-{{epoch:02d}}-{{val/loss:.4f}}",
        monitor='val/loss' if val_loader else 'train/loss',
        mode='min',
        save_top_k=checkpoint_config['save_top_k'],
        save_last=True,
        verbose=True,
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
        save_dir=log_config['dir'],
        name=log_config['experiment_name'],
    )
    
    # Setup trainer
    max_epochs = train_config['max_epochs']

    print(f"\nTraining configuration:")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {train_config['learning_rate']}")
    print(f"  Batch size: {data_config['batch_size']}")
    print(f"  Accelerator: {trainer_config['accelerator']}")
    print(f"  Devices: {trainer_config['devices']}")
    print(f"  Precision: {trainer_config['precision']}")
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        # pass through Trainer kwargs from config (load_config applies defaults)
        accelerator=trainer_config['accelerator'],
        devices=trainer_config['devices'],
        precision=trainer_config['precision'],
        gradient_clip_val=trainer_config['gradient_clip_val'],
        accumulate_grad_batches=trainer_config['accumulate_grad_batches'],
        log_every_n_steps=trainer_config['log_every_n_steps'],
        deterministic=train_config.get('deterministic', False),
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
