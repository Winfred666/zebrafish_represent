"""
PyTorch Lightning module for training 3D UNet on masked volume reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from typing import Dict, Optional, List
from pathlib import Path
import sys

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent.parent))
from model.unet3d import UNet3D


class FishModule(L.LightningModule):
    """
    Lightning module for training UNet on masked volume reconstruction task.
    
    This implements a self-supervised pretext task where the model learns to
    reconstruct masked regions of 3D volumes.
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: List[int] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 trilinear: bool = False,
                 use_batchnorm: bool = True,
                 loss_type: str = "mse",
                 **kwargs):
        """
        Initialize the FishModule.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Feature channels at each U-Net level
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            trilinear: Use trilinear upsampling in U-Net
            use_batchnorm: Use batch normalization in U-Net
            loss_type: Loss function type ("mse", "l1", "smooth_l1")
        """
        super().__init__()
        self.save_hyperparameters()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        # Initialize U-Net model
        self.model = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            trilinear=trilinear,
            use_batchnorm=use_batchnorm
        )
        
        # Loss function
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch.
        
        Args:
            batch: Dictionary containing 'input', 'target', and 'mask'
            
        Returns:
            Dictionary containing loss components
        """
        input_volume = batch['input']
        target_volume = batch['target']
        mask = batch['mask']
        
        # Forward pass
        pred = self(input_volume)
        
        # Compute reconstruction loss
        recon_loss = self.criterion(pred, target_volume)
        
        # Compute masked region loss (focus on reconstructing masked areas)
        masked_pred = pred * mask
        masked_target = target_volume * mask
        masked_loss = self.criterion(masked_pred, masked_target)
        
        # Compute unmasked region loss (preserve visible areas)
        unmasked_pred = pred * (1 - mask)
        unmasked_target = target_volume * (1 - mask)
        unmasked_loss = self.criterion(unmasked_pred, unmasked_target)
        
        # Total loss is weighted combination
        total_loss = recon_loss + 2.0 * masked_loss  # Weight masked loss higher
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'masked_loss': masked_loss,
            'unmasked_loss': unmasked_loss
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step."""
        losses = self.compute_loss(batch)
        
        # Log losses
        self.log('train/loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/recon_loss', losses['recon_loss'], on_step=False, on_epoch=True)
        self.log('train/masked_loss', losses['masked_loss'], on_step=False, on_epoch=True)
        self.log('train/unmasked_loss', losses['unmasked_loss'], on_step=False, on_epoch=True)
        
        return losses['loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        losses = self.compute_loss(batch)
        
        # Log losses
        self.log('val/loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recon_loss', losses['recon_loss'], on_step=False, on_epoch=True)
        self.log('val/masked_loss', losses['masked_loss'], on_step=False, on_epoch=True)
        self.log('val/unmasked_loss', losses['unmasked_loss'], on_step=False, on_epoch=True)
        
        return losses['loss']
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-7
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', current_lr, on_epoch=True)


class PretextDataset(torch.utils.data.Dataset):
    """
    Dataset for loading pretext training data from .pt files.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to .pt file containing list of samples
        """
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloaders(train_path: str,
                      val_path: Optional[str] = None,
                      batch_size: int = 4,
                      num_workers: int = 4) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_path: Path to training .pt file
        val_path: Path to validation .pt file (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary containing train and val dataloaders
    """
    train_dataset = PretextDataset(train_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    dataloaders = {'train': train_loader}
    
    if val_path is not None:
        val_dataset = PretextDataset(val_path)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )
        dataloaders['val'] = val_loader
    
    return dataloaders
