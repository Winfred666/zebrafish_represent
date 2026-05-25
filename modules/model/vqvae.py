"""3D VQ-VAE with encoder, codebook, and decoder for volume compression.

Ported from ``result/refer/3D-MedDiffusion/AutoEncoder/model/PatchVolume.py``.
Uses ``einops.rearrange`` instead of the reference ``shift_dim`` utility.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def rearrange_many(tensors: list[torch.Tensor], pattern: str, **axes_lengths: int) -> list[torch.Tensor]:
    """Apply ``rearrange`` to each tensor in a list."""
    return [rearrange(t, pattern, **axes_lengths) for t in tensors]

from modules.block.codebook import Codebook


# ---------------------------------------------------------------------------
# normalization helper
# ---------------------------------------------------------------------------

def Normalize(in_channels: int, norm_type: str = "group", num_groups: int = 32) -> nn.Module:
    assert norm_type in ("group", "batch")
    if norm_type == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    return nn.BatchNorm3d(in_channels)


# ---------------------------------------------------------------------------
# residual blocks (reference ResBlockX / ResBlockXY)
# ---------------------------------------------------------------------------

class ResBlockX(nn.Module):
    """ResBlock without channel change — identity skip."""

    def __init__(self, in_channels: int, out_channels: int | None = None,
                 dropout: float = 0.0, norm_type: str = "group", num_groups: int = 32):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class ResBlockXY(nn.Module):
    """ResBlock with channel change — 1x1x1 conv skip."""

    def __init__(self, in_channels: int, out_channels: int | None = None,
                 dropout: float = 0.0, norm_type: str = "group", num_groups: int = 32):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.norm2 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_conv(x)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + residual


# ---------------------------------------------------------------------------
# 3D attention block
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """3D spatial self-attention block with GroupNorm pre-norm and residual."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32,
                 norm_type: str = "group", num_groups: int = 32):
        super().__init__()
        self.norm = Normalize(dim, norm_type=norm_type, num_groups=num_groups)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x_norm = self.norm(x)
        x_norm = rearrange(x_norm, "b c d h w -> b (d h w) c").contiguous()
        qkv = self.to_qkv(x_norm).chunk(3, dim=2)
        q, k, v = rearrange_many(qkv, "b n (h c) -> b h n c", h=self.heads)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = rearrange(out, "b h (d hh w) c -> b (h c) d hh w", d=d, hh=h, w=w).contiguous()
        return self.to_out(out) + x


# ---------------------------------------------------------------------------
# upsampling
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_trans = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=4,
                                             stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_trans(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """3D CNN encoder with progressive downsampling.

    Parameters
    ----------
    n_hiddens : int
        Base channel count (doubled at each level).
    downsample : tuple[int, int, int]
        Total downsample ratios per axis, e.g. ``(8, 8, 8)``.
    image_channel : int
        Input channels (default 1).
    norm_type : str
        ``"group"`` or ``"batch"``.
    num_groups : int
        Groups for GroupNorm.
    embedding_dim : int
        Output channels (codebook embedding dimension).
    """

    def __init__(self, n_hiddens: int, downsample: tuple[int, int, int],
                 image_channel: int = 1, norm_type: str = "group",
                 num_groups: int = 32, embedding_dim: int = 8):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        max_ds = n_times_downsample.max()

        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(image_channel, n_hiddens, kernel_size=3, stride=1, padding=1)

        channels = [n_hiddens * (2 ** i) for i in range(max_ds)]
        channels = channels + [channels[-1]]

        self.conv_blocks = nn.ModuleList()
        for i in range(max_ds + 1):
            in_ch = channels[0] if i == 0 else channels[i - 1]
            out_ch = channels[i]
            stride = tuple(2 if d > 0 else 1 for d in n_times_downsample)

            block = nn.Module()
            if in_ch != out_ch:
                block.res1 = ResBlockXY(in_ch, out_ch, norm_type=norm_type, num_groups=num_groups)
            else:
                block.res1 = ResBlockX(in_ch, out_ch, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlockX(out_ch, out_ch, norm_type=norm_type, num_groups=num_groups)
            if i != max_ds:
                block.down = nn.Conv3d(out_ch, out_ch, kernel_size=4, stride=stride, padding=1)
            else:
                block.down = nn.Identity()
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        out_channels = channels[-1]
        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlockX(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
        self.mid_block.attn = AttentionBlock(out_channels, heads=4, norm_type=norm_type, num_groups=num_groups)
        self.mid_block.res2 = ResBlockX(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv3d(out_channels, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.res1(h)
            h = block.res2(h)
            h = block.down(h)
        h = self.mid_block.res1(h)
        h = self.mid_block.attn(h)
        h = self.mid_block.res2(h)
        h = self.final_block(h)
        return h


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """3D CNN decoder with progressive upsampling (mirror of Encoder)."""

    def __init__(self, n_hiddens: int, upsample: tuple[int, int, int],
                 image_channel: int = 1, norm_type: str = "group",
                 num_groups: int = 32, embedding_dim: int = 8):
        super().__init__()
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        channels = [n_hiddens * (2 ** i) for i in range(max_us)]
        channels = channels + [channels[-1]]
        channels.reverse()

        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(embedding_dim, channels[0], kernel_size=3, stride=1, padding=1)

        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlockX(channels[0], channels[0], norm_type=norm_type, num_groups=num_groups)
        self.mid_block.attn = AttentionBlock(channels[0], heads=4, norm_type=norm_type, num_groups=num_groups)
        self.mid_block.res2 = ResBlockX(channels[0], channels[0], norm_type=norm_type, num_groups=num_groups)

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us + 1):
            in_ch = channels[0] if i == 0 else channels[i - 1]
            out_ch = channels[i]
            us = tuple(2 if d > 0 else 1 for d in n_times_upsample)

            block = nn.Module()
            if in_ch != out_ch:
                block.res1 = ResBlockXY(in_ch, out_ch, norm_type=norm_type, num_groups=num_groups)
            else:
                block.res1 = ResBlockX(in_ch, out_ch, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlockX(out_ch, out_ch, norm_type=norm_type, num_groups=num_groups)
            if i != max_us:
                block.up = Upsample(out_ch)
            else:
                block.up = nn.Identity()
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        out_channels = channels[-1]
        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv3d(out_channels, image_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_first(x)
        h = self.mid_block.res1(h)
        h = self.mid_block.attn(h)
        h = self.mid_block.res2(h)
        for block in self.conv_blocks:
            h = block.res1(h)
            h = block.res2(h)
            h = block.up(h)
        h = self.final_block(h)
        return h


# ---------------------------------------------------------------------------
# VQ-VAE wrapper
# ---------------------------------------------------------------------------

class VQVAE(nn.Module):
    """Full VQ-VAE: Encoder -> pre-vq conv -> Codebook -> post-vq conv -> Decoder.

    Supports two modes:
    - **full-volume** (stage 1): encode/decode the entire volume at once.
    - **patch-based** (stage 2): unfold volume into patches, encode each,
      reassemble, then decode.

    Parameters
    ----------
    n_hiddens : int
        Base channel count.
    downsample : tuple[int, int, int]
        Per-axis downsample ratios, e.g. ``(8, 8, 8)``.
    image_channel : int
        Input/output channels.
    embedding_dim : int
        Codebook embedding dimension.
    n_codes : int
        Codebook size.
    norm_type : str
        ``"group"`` or ``"batch"``.
    num_groups : int
        Groups for GroupNorm.
    no_random_restart : bool
        Disable codebook random restart.
    restart_thres : float
        Usage threshold for random restart.
    patch_size : int
        Patch size for stage-2 patch-based encoding (default 64).
    """

    def __init__(
        self,
        n_hiddens: int = 64,
        downsample: tuple[int, int, int] = (8, 8, 8),
        image_channel: int = 1,
        embedding_dim: int = 8,
        n_codes: int = 512,
        norm_type: str = "group",
        num_groups: int = 32,
        no_random_restart: bool = False,
        restart_thres: float = 1.0,
        patch_size: int = 64,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample = downsample
        self.patch_size = patch_size

        self.encoder = Encoder(
            n_hiddens=n_hiddens, downsample=downsample,
            image_channel=image_channel, norm_type=norm_type,
            num_groups=num_groups, embedding_dim=embedding_dim,
        )
        self.decoder = Decoder(
            n_hiddens=n_hiddens, upsample=downsample,
            image_channel=image_channel, norm_type=norm_type,
            num_groups=num_groups, embedding_dim=embedding_dim,
        )
        self.pre_vq_conv = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1, stride=1)
        self.post_vq_conv = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1, stride=1)
        self.codebook = Codebook(
            n_codes=n_codes, embedding_dim=embedding_dim,
            no_random_restart=no_random_restart, restart_thres=restart_thres,
        )

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Encode volume to latent codes.

        Returns:
            If ``quantize=True``: ``(B, D//ds, H//ds, W//ds)`` integer code indices.
            If ``quantize=False``: ``(B, embedding_dim, D//ds, H//ds, W//ds)`` continuous latent.
        """
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            return vq_output["encodings"]
        return h

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent codes back to volume space.

        Args:
            latent: ``(B, D_lat, H_lat, W_lat)`` integer code indices
                OR ``(B, C, D_lat, H_lat, W_lat)`` continuous features.
        """
        if latent.dim() == 4:
            # integer code indices -> embed
            h = F.embedding(latent, self.codebook.embeddings)
            h = rearrange(h, "b d h w c -> b c d h w")
        else:
            # continuous features -> quantize first
            vq_output = self.codebook(latent)
            h = vq_output["embeddings"]
        h = self.post_vq_conv(h)
        return self.decoder(h)

    # ------------------------------------------------------------------
    # patch-based encode (stage 2)
    # ------------------------------------------------------------------

    def patch_encode(self, x: torch.Tensor, quantize: bool = False) -> torch.Tensor:
        """Unfold volume into patches, encode each, reassemble into latent grid.

        Used in stage 2 where the full volume exceeds GPU memory.
        """
        b = x.shape[0]
        ps = self.patch_size
        s1, s2, s3 = x.shape[-3], x.shape[-2], x.shape[-1]

        x = x.unfold(2, ps, ps).unfold(3, ps, ps).unfold(4, ps, ps)
        x = rearrange(x, "b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w")
        h = self.pre_vq_conv(self.encoder(x))

        if quantize:
            vq_output = self.codebook(h)
            embeddings = vq_output["embeddings"]
        else:
            embeddings = h

        embeddings = rearrange(embeddings, "(b p) c d h w -> b p c d h w", b=b)
        embeddings = rearrange(
            embeddings, "b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)",
            p1=s1 // ps, p2=s2 // ps, p3=s3 // ps,
        )
        return embeddings

    # ------------------------------------------------------------------
    # one-step reconstruction (for validation visualization)
    # ------------------------------------------------------------------

    def one_step_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode -> quantize -> decode in one step. Used for val reconstruction."""
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        return self.decoder(self.post_vq_conv(vq_output["embeddings"]))

    # ------------------------------------------------------------------
    # forward (for training - returns reconstruction + vq output)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Full forward pass: encode -> quantize -> decode.

        Returns:
            ``(x_recon, vq_output)`` where ``vq_output`` contains
            ``embeddings``, ``encodings``, ``commitment_loss``, ``perplexity``.
        """
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))
        return x_recon, vq_output

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
