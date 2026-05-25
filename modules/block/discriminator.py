"""3D patch discriminator for VQ-VAE GAN training.

Ported from ``result/refer/3D-MedDiffusion/AutoEncoder/model/PatchVolume.py``.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn


class NLayerDiscriminator3D(nn.Module):
    """Multi-layer 3D convolutional discriminator with intermediate features.

    Used for adversarial training of the VQ-VAE decoder.  Returns both the
    final logits and a list of intermediate feature maps (for feature-matching
    loss in the generator).
    """

    def __init__(
        self,
        input_nc: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        use_sigmoid: bool = False,
        getIntermFeat: bool = True,
    ):
        super().__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    nn.BatchNorm3d(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                nn.BatchNorm3d(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [
            [nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]
        ]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, x):
        if self.getIntermFeat:
            res = [x]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(x), None
