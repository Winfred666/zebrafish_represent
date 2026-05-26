"""BiFlowNet: dual-path (U-Net + intra-patch DiT flow) diffusion model.

Ported from ``result/refer/3D-MedDiffusion/ddpm/BiFlowNet.py``.
Reuses existing DiT blocks and the DDPM training framework as-is.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from modules.block.common import modulate
from modules.block.decoder import FinalLayer3D
from modules.block.dit import DiTBlock3D
from modules.block.encoder import ConvPatchTokenizer3D
from modules.block.unet import (
    Block,
    Downsample,
    ResnetBlock,
    SinusoidalPosEmb,
    SpatialAttentionBlock,
    SpatialLayerNorm,
    Upsample,
)
from modules.model.base import BaseVolumeModel


# ---------------------------------------------------------------------------
# helpers (ported from reference BiFlowNet.py)
# ---------------------------------------------------------------------------

def _default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


# ---------------------------------------------------------------------------
# frozen 3D sinusoidal position embedding  (reference: get_3d_sincos_pos_embed)
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_3d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    emb_x = _get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_y = _get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_z = _get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    return np.concatenate([emb_x, emb_y, emb_z], axis=1)


def _get_3d_sincos_pos_embed(embed_dim: int, grid_size: tuple) -> np.ndarray:
    grid_x = np.arange(grid_size[0], dtype=np.float32)
    grid_y = np.arange(grid_size[1], dtype=np.float32)
    grid_z = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    return _get_3d_sincos_pos_embed_from_grid(embed_dim, grid)


# ---------------------------------------------------------------------------
# lightweight residual / pre-norm wrappers
# ---------------------------------------------------------------------------

class _Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.fn(x, *args, **kwargs) + x


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = SpatialLayerNorm(dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


# ---------------------------------------------------------------------------
# skip-connection DiT block wrapper
# ---------------------------------------------------------------------------

class _SkipDiTBlock(nn.Module):
    """DiTBlock3D wrapper that concatenates skip tokens before the block.

    The reference DiTBlock has a built-in ``skip_linear``.  We achieve the
    same effect without modifying ``DiTBlock3D`` by concatenating along the
    token dimension and projecting back to ``hidden_size`` before the block.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        self.dit_block = DiTBlock3D(
            hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio
        )

    def forward(self, x: Tensor, condition: Tensor, skip: Tensor) -> Tensor:
        x = self.skip_linear(torch.cat([x, skip], dim=-1))
        return self.dit_block(x, condition)


# ---------------------------------------------------------------------------
# BiFlowNet
# ---------------------------------------------------------------------------

class BiFlowNet(BaseVolumeModel):
    """Dual-path diffusion model: 3D U-Net + intra-patch DiT flow with feature fusion.

    The DiT path operates on sub-volumes (extracted via ``unfold``), processes
    them through DiT blocks with frozen sin-cos positional embeddings, then
    unpatchifies and injects the resulting feature maps into the U-Net path
    at matching resolutions.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        input_size: tuple[int, int, int] = (32, 32, 32),
        dim: int = 64,
        dim_mults: tuple[int, ...] = (1, 1, 2, 4, 8),
        sub_volume_size: tuple[int, int, int] = (8, 8, 8),
        patch_size: int = 2,
        attn_heads: int = 8,
        init_dim: int | None = None,
        init_kernel_size: int = 3,
        use_sparse_linear_attn: tuple[int, ...] = (0, 0, 0, 1, 1),
        resnet_groups: int = 24,
        dit_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_mid_dit: int = 1,
        cond_classes: int | None = None,
        res_condition: bool = True,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_size = tuple(int(v) for v in input_size)
        self.cond_classes = cond_classes
        self.res_condition = res_condition

        out_dim = 2 * out_channels if learn_sigma else out_channels
        self.dim = dim
        init_dim = _default(init_dim, dim)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            in_channels, init_dim,
            kernel_size=init_kernel_size,
            padding=init_padding,
        )

        # --- compute U-Net channel progression ---
        dims = [init_dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.feature_fusion = sum(1 for d_in, d_out in in_out if d_in == d_out)

        # --- time embedding (reference: time_mlp) ---
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- optional class / resolution conditioning ---
        if cond_classes is not None:
            self.cond_emb = nn.Embedding(cond_classes, time_dim)
        if res_condition:
            self.res_mlp = nn.Sequential(
                nn.Linear(3, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
            time_dim = 2 * time_dim  # doubled when res_condition is active

        # --- project condition for DiTBlock3D compatibility ---
        # (reference DiTBlock expects 4*hidden_size*2 dims; our DiTBlock3D
        #  expects hidden_size.  Project through a learned linear layer.)
        self.dit_cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim),
        )

        # --- intra-patch DiT flow (tokenizer + pos embed + blocks) ---
        self.sub_volume_size = sub_volume_size
        self.patch_size = patch_size
        self.x_embedder = ConvPatchTokenizer3D(
            in_channels=in_channels,
            embed_dim=dim,
            patch_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
        )
        num_patches = self.x_embedder.compute_grid_size(sub_volume_size)
        num_patches = num_patches[0] * num_patches[1] * num_patches[2]
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, dim), requires_grad=False
        )

        # IntraPatchFlow_input: feature_fusion x [DiTBlock3D, FinalLayer3D]
        self.intra_patch_input = nn.ModuleList()
        for _ in range(self.feature_fusion):
            self.intra_patch_input.append(nn.ModuleList([
                DiTBlock3D(hidden_size=dim, num_heads=dit_num_heads, mlp_ratio=mlp_ratio),
                FinalLayer3D(dim, patch_size ** 3, dim),
            ]))

        # IntraPatchFlow_mid: num_mid_dit x DiTBlock3D
        self.intra_patch_mid = nn.ModuleList([
            DiTBlock3D(hidden_size=dim, num_heads=dit_num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_mid_dit)
        ])

        # IntraPatchFlow_output: feature_fusion x [_SkipDiTBlock, FinalLayer3D]
        self.intra_patch_output = nn.ModuleList()
        for _ in range(self.feature_fusion):
            self.intra_patch_output.append(nn.ModuleList([
                _SkipDiTBlock(hidden_size=dim, num_heads=dit_num_heads, mlp_ratio=mlp_ratio),
                FinalLayer3D(dim, patch_size ** 3, dim),
            ]))

        # --- U-Net down path ---
        num_resolutions = len(in_out)
        block_klass_cond = lambda din, dout: ResnetBlock(
            din, dout, time_emb_dim=time_dim, groups=resnet_groups
        )

        self.downs = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            is_first = ind < self.feature_fusion - 1
            use_attn = bool(use_sparse_linear_attn[ind])
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                _Residual(_PreNorm(dim_out, SpatialAttentionBlock(dim_out, heads=attn_heads)))
                if use_attn else nn.Identity(),
                block_klass_cond(dim_out, dim_out),
                _Residual(_PreNorm(dim_out, SpatialAttentionBlock(dim_out, heads=attn_heads)))
                if use_attn else nn.Identity(),
                Downsample(dim_out) if (not is_last and not is_first) else nn.Identity(),
            ]))

        # --- U-Net middle ---
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        self.mid_spatial_attn = _Residual(
            _PreNorm(mid_dim, SpatialAttentionBlock(mid_dim, heads=attn_heads))
        )
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        # --- U-Net up path ---
        self.ups = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 2)
            use_attn = bool(use_sparse_linear_attn[len(in_out) - ind - 1])
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_out),
                _Residual(_PreNorm(dim_out, SpatialAttentionBlock(dim_out, heads=attn_heads)))
                if use_attn else nn.Identity(),
                block_klass_cond(dim_out * 2, dim_in),
                _Residual(_PreNorm(dim_in, SpatialAttentionBlock(dim_in, heads=attn_heads)))
                if use_attn else nn.Identity(),
                Upsample(dim_in) if not is_last else nn.Identity(),
            ]))

        # --- final conv ---
        self.final_conv = nn.Sequential(
            ResnetBlock(dim * 2, dim, groups=resnet_groups),
            nn.Conv3d(dim, out_dim, 1),
        )

        self.initialize_weights()

    # ------------------------------------------------------------------
    # weight initialisation  (reference: initialize_weights)
    # ------------------------------------------------------------------

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # frozen sin-cos pos_embed
        grid_size = (
            self.sub_volume_size[0] // self.patch_size,
            self.sub_volume_size[1] // self.patch_size,
            self.sub_volume_size[2] // self.patch_size,
        )
        pos_embed = _get_3d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.as_tensor(pos_embed).float().unsqueeze(0))

        # x_embedder (ConvPatchTokenizer3D) init
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        # zero-init adaLN modulation in DiT blocks + FinalLayers
        for block_list in (*self.intra_patch_input, *self.intra_patch_output):
            for blk in block_list:
                if isinstance(blk, _SkipDiTBlock):
                    nn.init.constant_(blk.dit_block.ada_ln_modulation[-1].weight, 0)
                    nn.init.constant_(blk.dit_block.ada_ln_modulation[-1].bias, 0)
                elif isinstance(blk, DiTBlock3D):
                    nn.init.constant_(blk.ada_ln_modulation[-1].weight, 0)
                    nn.init.constant_(blk.ada_ln_modulation[-1].bias, 0)
                if isinstance(blk, FinalLayer3D):
                    nn.init.constant_(blk.linear.weight, 0)
                    nn.init.constant_(blk.linear.bias, 0)

        for blk in self.intra_patch_mid:
            nn.init.constant_(blk.ada_ln_modulation[-1].weight, 0)
            nn.init.constant_(blk.ada_ln_modulation[-1].bias, 0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _unpatchify_voxels(self, x: Tensor) -> Tensor:
        """(N, T, patch^3 * C) -> (N, C, X, Y, Z)  (reference: unpatchify_voxels)."""
        c = self.dim
        p = self.patch_size
        sv = self.sub_volume_size
        x_g, y_g, z_g = sv[0] // p, sv[1] // p, sv[2] // p
        x = x.reshape(x.shape[0], x_g, y_g, z_g, p, p, p, c)
        x = torch.einsum("nxyzpqrc->ncxpyqzr", x)
        return x.reshape(x.shape[0], c, x_g * p, y_g * p, z_g * p)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        *,
        pos_idx: Tensor | None = None,
        validate: bool = False,
        y: Tensor | None = None,
        res: Tensor | None = None,
    ) -> Tensor:
        """Predict noise (epsilon) for the given noisy volume.

        Args:
            x: (B, C, D, H, W) noisy volume
            timesteps: (B,) normalised timesteps in [0, 1]
            pos_idx: unused (accepted for BaseVolumeModel compatibility)
            validate: unused
            y: optional class labels (B,) for conditional generation
            res: optional resolution tensor (B, 3) for resolution conditioning
        """
        b = x.shape[0]
        in_sz = self.input_size
        sv = self.sub_volume_size

        # --- extract sub-volumes for DiT path ---
        x_intra = x.unfold(2, sv[0], sv[0]).unfold(3, sv[1], sv[1]).unfold(4, sv[2], sv[2])
        p1, p2, p3 = x_intra.size(2), x_intra.size(3), x_intra.size(4)
        x_intra = rearrange(x_intra, "b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w")

        # --- U-Net initial conv ---
        x_u = self.init_conv(x)
        r = x_u.clone()

        # --- time embedding ---
        t = self.time_mlp(timesteps) if self.time_mlp is not None else None
        c_dim = t.shape[-1]
        t_dit = t.unsqueeze(1).repeat(1, p1 * p2 * p3, 1).view(-1, c_dim)

        if self.cond_classes is not None:
            cond_emb = self.cond_emb(y)
            cond_emb_dit = cond_emb.unsqueeze(1).repeat(1, p1 * p2 * p3, 1).view(-1, c_dim)
            t = t + cond_emb
            t_dit = t_dit + cond_emb_dit

        if self.res_condition and res is not None:
            if res.ndim == 1:
                res = res.unsqueeze(0)
            res_cond_emb = self.res_mlp(res)
            t = torch.cat((t, res_cond_emb), dim=1)
            res_cond_emb_dit = res_cond_emb.unsqueeze(1).repeat(1, p1 * p2 * p3, 1).view(-1, c_dim)
            t_dit = torch.cat((t_dit, res_cond_emb_dit), dim=1)

        # --- project condition for DiTBlock3D compatibility ---
        t_dit = self.dit_cond_proj(t_dit)

        # --- DiT path ---
        x_intra = self.x_embedder(x_intra)
        x_intra = x_intra + self.pos_embed

        h_dit_stack: list[Tensor] = []
        h_unet: list[Tensor] = []

        for dit_block, final_layer in self.intra_patch_input:
            x_intra = dit_block(x_intra, t_dit)
            h_dit_stack.append(x_intra)
            unet_feat = self._unpatchify_voxels(final_layer(x_intra, t_dit))
            unet_feat = rearrange(unet_feat, "(b p) c d h w -> b p c d h w", b=b)
            unet_feat = rearrange(
                unet_feat, "b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)",
                p1=in_sz[0] // sv[0], p2=in_sz[1] // sv[1], p3=in_sz[2] // sv[2],
            )
            h_unet.append(unet_feat)

        for dit_block in self.intra_patch_mid:
            x_intra = dit_block(x_intra, t_dit)

        for skip_dit_block, final_layer in self.intra_patch_output:
            x_intra = skip_dit_block(x_intra, t_dit, h_dit_stack.pop())
            unet_feat = self._unpatchify_voxels(final_layer(x_intra, t_dit))
            unet_feat = rearrange(unet_feat, "(b p) c d h w -> b p c d h w", b=b)
            unet_feat = rearrange(
                unet_feat, "b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)",
                p1=in_sz[0] // sv[0], p2=in_sz[1] // sv[1], p3=in_sz[2] // sv[2],
            )
            h_unet.append(unet_feat)

        # --- U-Net down path ---
        h_skips: list[Tensor] = []
        for idx, (block1, attn1, block2, attn2, down) in enumerate(self.downs):
            if idx < self.feature_fusion:
                x_u = x_u + h_unet.pop(0)
            x_u = block1(x_u, t)
            if not isinstance(attn1, nn.Identity):
                x_u = attn1(x_u)
            h_skips.append(x_u)
            x_u = block2(x_u, t)
            if not isinstance(attn2, nn.Identity):
                x_u = attn2(x_u)
            h_skips.append(x_u)
            x_u = down(x_u)

        # --- U-Net middle ---
        x_u = self.mid_block1(x_u, t)
        x_u = self.mid_spatial_attn(x_u)
        x_u = self.mid_block2(x_u, t)

        # --- U-Net up path ---
        for idx, (block1, attn1, block2, attn2, up) in enumerate(self.ups):
            if len(self.ups) - idx <= 2:
                x_u = x_u + h_unet.pop(0)
            x_u = torch.cat((x_u, h_skips.pop()), dim=1)
            x_u = block1(x_u, t)
            if not isinstance(attn1, nn.Identity):
                x_u = attn1(x_u)
            x_u = torch.cat((x_u, h_skips.pop()), dim=1)
            x_u = block2(x_u, t)
            if not isinstance(attn2, nn.Identity):
                x_u = attn2(x_u)
            x_u = up(x_u)

        # --- final ---
        x_u = torch.cat((x_u, r), dim=1)
        return self.final_conv(x_u)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
