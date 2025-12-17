import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange, reduce

from functools import partial
from ..core import register
from .dinov3_vision_transformer import *


__all__ = ['DINOv3SGAs']


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4

class SemanticGuidingModule(nn.Module):
    def __init__(self, semantic_channels: int):
        super().__init__()
        self.attention_generator = nn.Sequential(
            nn.Conv2d(semantic_channels, semantic_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.SyncBatchNorm(semantic_channels // 4),
            nn.GELU(),
            nn.Conv2d(semantic_channels // 4, 1, kernel_size=1)
        )

    def forward(self, semantic_feature: torch.Tensor, target_shapes: List[torch.Size]) -> List[torch.Tensor]:
        attention_score_map = self.attention_generator(semantic_feature)
        attention_maps = []
        for shape in target_shapes:
            resized_map = F.interpolate(attention_score_map, size=shape[2:], mode='bilinear', align_corners=False)
            attention_map = torch.sigmoid(resized_map)
            attention_maps.append(attention_map)
        return attention_maps
    


@register()
class DINOv3SGAs(nn.Module):
    def __init__(
        self,
        name: str = 'dinov3_small',
        weights_path: str = None,
        interaction_indexes: List[int] = [9, 10, 11],
        finetune: bool = True,
        patch_size: int = 16,
        conv_inplane: int = 16,
        hidden_dim: int = 256,
        embed_dim: int = 384,
    ):
        super().__init__()

        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        vit_constructors = {
            'dinov3_small': vit_small,
            'dinov3_base': vit_base,
            'dinov3_large': vit_large,
            'dinov3_huge': vit_huge2,
            'dinov3_7b': vit_7b
        }
        
        model_size_key = name.replace('dinov3_', '')
        
        if name not in vit_constructors:
            raise NotImplementedError(f"Model name '{name}' is not supported.")
        vit_builder = vit_constructors[name]
        print(f"Building DinoVisionTransformer with '{name}' configuration...")
        self.backbone = vit_builder(patch_size=self.patch_size, pretrained=weights_path)
        
        self.embed_dim = self.backbone.embed_dim


        if not finetune:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            print("DinoVisionTransformer backbone parameters are frozen.")

        print(f"Using SpatialPriorModulev2 Adapter with inplanes={conv_inplane}")

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.context_aggregator = SemanticGuidingModule(semantic_channels=embed_dim)

        adapter_dims = [conv_inplane*2, conv_inplane*4, conv_inplane*4]
        self.fusion_layers = nn.ModuleList()
        for i in range(len(interaction_indexes)):
            self.fusion_layers.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim + adapter_dims[i], hidden_dim, kernel_size=1, bias=False),
                    nn.SyncBatchNorm(hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                    nn.SyncBatchNorm(hidden_dim)
                )
            )

    def forward(self, x: torch.Tensor):
        bs, _, H_img, W_img = x.shape
        H_feat, W_feat = H_img // self.patch_size, W_img // self.patch_size


        all_layers_outputs = self.backbone.get_intermediate_layers(
            x, n=self.interaction_indexes, reshape=True, return_class_token=True
        )
        embed_dim = self.embed_dim
        sem_feats_2d = []
        for layer_output in all_layers_outputs:
            patch_tokens, _ = layer_output 
            sem_feat_2d = patch_tokens
            sem_feats_2d.append(sem_feat_2d)

        detail_feats = self.sta(x) # (c2, c3, c4)
        
        context_maps = self.context_aggregator(sem_feats_2d[-1], [f.shape for f in detail_feats])
        
        outputs = []
        target_scales = [H_feat * 2, H_feat, H_feat // 2] 
        
        for i in range(len(all_layers_outputs)):
            sem_feat_2d = sem_feats_2d[i]
            detail_feat = detail_feats[i]
            context_map = context_maps[i]
            fusion_layer = self.fusion_layers[i]
            
            target_size = (int(target_scales[i]), int(target_scales[i]))
            aligned_sem_feat = F.interpolate(sem_feat_2d, size=target_size, mode="bilinear", align_corners=False)
            
            enhanced_detail_feat = detail_feat * context_map
            
            fused_feat = torch.cat([aligned_sem_feat, enhanced_detail_feat], dim=1)
            output = fusion_layer(fused_feat)
            
            outputs.append(output)
            
        return tuple(outputs)

