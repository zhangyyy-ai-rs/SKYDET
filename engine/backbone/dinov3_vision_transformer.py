"""SKYDET-compatible DINOv3 ViT backbones using the bundled official implementation."""

from pathlib import Path
from typing import Optional

import torch

from ..core import register
from ..dinov3.dinov3.models.vision_transformer import (
    DinoVisionTransformer as OfficialDinoVisionTransformer,
)


_MODEL_SPECS = {
    "dinov3_vits16": dict(embed_dim=384, depth=12, num_heads=6, ffn_ratio=4, qkv_bias=True,
                           drop_path_rate=0.0, ffn_layer="mlp"),
    "dinov3_vits16plus": dict(embed_dim=384, depth=12, num_heads=6, ffn_ratio=6, qkv_bias=True,
                               drop_path_rate=0.0, ffn_layer="swiglu"),
    "dinov3_vitb16": dict(embed_dim=768, depth=12, num_heads=12, ffn_ratio=4, qkv_bias=True,
                           drop_path_rate=0.0, ffn_layer="mlp"),
    "dinov3_vitl16": dict(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4, qkv_bias=True,
                           drop_path_rate=0.0, ffn_layer="mlp"),
    "dinov3_vitl16plus": dict(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=6, qkv_bias=True,
                               drop_path_rate=0.0, ffn_layer="swiglu"),
    "dinov3_vith16plus": dict(embed_dim=1280, depth=32, num_heads=20, ffn_ratio=6, qkv_bias=True,
                               drop_path_rate=0.0, ffn_layer="swiglu"),
    "dinov3_vit7b16": dict(embed_dim=4096, depth=40, num_heads=32, ffn_ratio=3, qkv_bias=False,
                            drop_path_rate=0.4, ffn_layer="swiglu64"),
}


def _select_model_name(pretrained, embed_dim, depth, num_heads, ffn_ratio):
    checkpoint_name = Path(pretrained).name.lower() if pretrained else ""
    for name in (
        "dinov3_vits16plus", "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16plus",
        "dinov3_vitl16", "dinov3_vith16plus", "dinov3_vit7b16",
    ):
        if name in checkpoint_name:
            return name
    requested = (embed_dim, depth, num_heads, float(ffn_ratio))
    for name, spec in _MODEL_SPECS.items():
        if requested == (spec["embed_dim"], spec["depth"], spec["num_heads"], float(spec["ffn_ratio"])):
            return name
    raise ValueError(
        "Unsupported DINOv3 ViT configuration: "
        f"embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, ffn_ratio={ffn_ratio}."
    )


@register()
class DinoVisionTransformer(OfficialDinoVisionTransformer):
    """Official DINOv3 architecture with the original SKYDET constructor/output API."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        pretrained: Optional[str] = None,
        finetune: bool = False,
        **kwargs,
    ):
        if patch_size != 16:
            raise ValueError("The bundled official DINOv3 checkpoints use patch_size=16.")
        model_name = _select_model_name(pretrained, embed_dim, depth, num_heads, ffn_ratio)
        spec = dict(_MODEL_SPECS[model_name])
        spec.update(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            layerscale_init=1.0e-5,
            norm_layer="layernormbf16",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )
        checkpoint_name = Path(pretrained).name.lower() if pretrained else ""
        if model_name == "dinov3_vit7b16" or "eadcf0ff" in checkpoint_name:
            spec["untie_global_and_local_cls_norm"] = True
        spec.update(kwargs)
        super().__init__(**spec)

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=True)
        else:
            self.init_weights()
        if not finetune:
            self.requires_grad_(False)

    def forward(self, x):
        return self.get_intermediate_layers(x, n=1, reshape=True)


def _build(model_name, patch_size=16, pretrained=None, finetune=False, **kwargs):
    spec = _MODEL_SPECS[model_name]
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=spec["embed_dim"],
        depth=spec["depth"],
        num_heads=spec["num_heads"],
        ffn_ratio=spec["ffn_ratio"],
        pretrained=pretrained,
        finetune=finetune,
        **kwargs,
    )


def vit_small(patch_size=16, **kwargs):
    return _build("dinov3_vits16", patch_size, **kwargs)


def vit_base(patch_size=16, **kwargs):
    return _build("dinov3_vitb16", patch_size, **kwargs)


def vit_large(patch_size=16, **kwargs):
    return _build("dinov3_vitl16", patch_size, **kwargs)


def vit_huge2(patch_size=16, **kwargs):
    return _build("dinov3_vith16plus", patch_size, **kwargs)


def vit_7b(patch_size=16, **kwargs):
    return _build("dinov3_vit7b16", patch_size, **kwargs)
