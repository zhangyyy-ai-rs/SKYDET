"""SKYDET-compatible DINOv3 ConvNeXt backbones using the bundled official implementation."""

from pathlib import Path
from typing import List, Optional

import torch

from ..core import register
from ..dinov3.dinov3.models.convnext import ConvNeXt as OfficialConvNeXt


_MODEL_SPECS = {
    "dinov3_convnext_tiny": ([3, 3, 9, 3], [96, 192, 384, 768]),
    "dinov3_convnext_small": ([3, 3, 27, 3], [96, 192, 384, 768]),
    "dinov3_convnext_base": ([3, 3, 27, 3], [128, 256, 512, 1024]),
    "dinov3_convnext_large": ([3, 3, 27, 3], [192, 384, 768, 1536]),
}


def _select_model_name(pretrained, depths, dims):
    checkpoint_name = Path(pretrained).name.lower() if pretrained else ""
    for name in _MODEL_SPECS:
        if name in checkpoint_name:
            return name
    requested = (list(depths), list(dims))
    for name, spec in _MODEL_SPECS.items():
        if requested == spec:
            return name
    raise ValueError(f"Unsupported DINOv3 ConvNeXt configuration: depths={depths}, dims={dims}.")


@register()
class ConvNeXt(OfficialConvNeXt):
    """Official DINOv3 architecture with the original SKYDET constructor/output API."""

    def __init__(
        self,
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1.0e-6,
        return_idx: List[int] = [1, 2, 3],
        pretrained: Optional[str] = None,
        finetune: bool = False,
        patch_size=None,
        **kwargs,
    ):
        if in_chans != 3:
            raise ValueError("The bundled official DINOv3 ConvNeXt checkpoints use in_chans=3.")
        model_name = _select_model_name(pretrained, depths, dims)
        official_depths, official_dims = _MODEL_SPECS[model_name]
        super().__init__(
            in_chans=3,
            depths=official_depths,
            dims=official_dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            patch_size=patch_size,
            **kwargs,
        )
        self.return_idx = list(return_idx)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=True)
        else:
            self.init_weights()
        if not finetune:
            self.requires_grad_(False)

    def forward(self, x):
        return self.get_intermediate_layers(x, self.return_idx, reshape=True)
