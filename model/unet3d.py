"""
Standard 3D U-Net implementation for volumetric data.
"""
import torch
import torch.nn as nn
from typing import List


class Conv3DBlock(nn.Module):
    """
    3D Convolutional block with BatchNorm and ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, use_batchnorm: bool = True):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class DoubleConv3D(nn.Module):
    """
    Two consecutive 3D convolutional blocks.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None,
                 use_batchnorm: bool = True):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            Conv3DBlock(in_channels, mid_channels, use_batchnorm=use_batchnorm),
            Conv3DBlock(mid_channels, out_channels, use_batchnorm=use_batchnorm)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    Downscaling with maxpool then double conv.
    """
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, use_batchnorm=use_batchnorm)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    Upscaling then double conv.
    """
    def __init__(self, in_channels: int, out_channels: int, trilinear: bool = False,
                 use_batchnorm: bool = True):
        super().__init__()
        
        # Use trilinear upsampling or transposed convolution
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2,
                                    use_batchnorm=use_batchnorm)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels, use_batchnorm=use_batchnorm)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatches due to pooling
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]
        
        x1 = nn.functional.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                                    diff_h // 2, diff_h - diff_h // 2,
                                    diff_d // 2, diff_d - diff_d // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Standard 3D U-Net architecture.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        features: List of feature channels at each level [64, 128, 256, 512]
        trilinear: Use trilinear upsampling instead of transposed convolution
        use_batchnorm: Use batch normalization
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: List[int] = None, trilinear: bool = False,
                 use_batchnorm: bool = True):
        super().__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.trilinear = trilinear
        
        # Input convolution
        self.inc = DoubleConv3D(in_channels, features[0], use_batchnorm=use_batchnorm)
        
        # Encoder (downsampling path)
        self.down_layers = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_layers.append(
                Down3D(features[i], features[i + 1], use_batchnorm=use_batchnorm)
            )
        
        # Decoder (upsampling path)
        self.up_layers = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up_layers.append(
                Up3D(features[i], features[i - 1], trilinear=trilinear,
                    use_batchnorm=use_batchnorm)
            )
        
        # Output convolution
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x = self.inc(x)
        skip_connections = [x]
        
        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)
        
        # Remove the last skip connection (bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Decoder
        for i, up in enumerate(self.up_layers):
            skip = skip_connections[-(i + 1)]
            x = up(x, skip)
        
        # Output
        x = self.outc(x)
        return x
    
    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_unet3d():
    """Test the UNet3D model with a sample input."""
    # Test with small input
    model = UNet3D(in_channels=1, out_channels=1, features=[32, 64, 128, 256])
    x = torch.randn(1, 1, 32, 64, 64)  # (B, C, D, H, W)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {model.get_num_params():,}")
    
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    print("UNet3D test passed!")


if __name__ == "__main__":
    test_unet3d()
