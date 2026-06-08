"""VQ-VAE codebook with EMA updates and straight-through gradient estimation."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Codebook(nn.Module):
    """Vector-quantisation codebook with exponential-moving-average updates.

    Ported from ``result/refer/3D-MedDiffusion/AutoEncoder/model/codebook.py``.
    """

    def __init__(
        self,
        n_codes: int,
        embedding_dim: int,
        no_random_restart: bool = False,
        restart_thres: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

    def _tile(self, x: torch.Tensor) -> torch.Tensor:
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / (ew ** 0.5)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    # performs data-dependent initialization of the codebook
    # Instead of starting with completely random values, this initializes the codebook vectors by randomly sampling directly from the continuous encoded inputs (z) of the very first training batch.
    def _init_embeddings(self, z: torch.Tensor) -> None:
        self._need_init = False
        flat_inputs = rearrange(z, "b c d h w -> (b d h w) c")
        y = self._tile(flat_inputs)
        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z: torch.Tensor) -> dict:
        # z: (B, C, D, H, W)
        if self._need_init and self.training:
            self._init_embeddings(z)

        flat_inputs = rearrange(z, "b c d h w -> (b d h w) c")

        # compute all distance to codes, use nearest neighbor lookup
        distances = (
            (flat_inputs ** 2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = rearrange(embeddings, "b d h w c -> b c d h w")

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        if self.training:
            # counts how many times each codebook vector was used in the current batch
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            # Exponential Moving Average (EMA) update of the codebook vectors
            
            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        # WARNING: straight-through estimator + stop-gradient for the discrete encoding operation
        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    def dictionary_lookup(self, encodings: torch.Tensor) -> torch.Tensor:
        return F.embedding(encodings, self.embeddings)
