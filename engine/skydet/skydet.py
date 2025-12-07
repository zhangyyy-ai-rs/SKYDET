"""
SKYDET: A Multi-Scale Attentive Detection Network from Foundation Models 
for Small Density-Aware Object in Remote Sensing Images
Copyright (c) 2025 The SKYDET Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['SKYDET']


@register()
class SKYDET(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
