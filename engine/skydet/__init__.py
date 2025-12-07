"""
SKYDET: A Multi-Scale Attentive Detection Network from Foundation Models 
for Small Density-Aware Object in Remote Sensing Images
Copyright (c) 2025 The SKYDET Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM (https://github.com/Intellindust-AI-Lab/DEIM)
Copyright (c) 2025 DEIM Authors. All Rights Reserved.
"""


from .skydet import SKYDET

from .matcher import HungarianMatcher
from .hybrid_encoder import HybridEncoder
from .dfine_decoder import DFINETransformer

from .postprocessor import PostProcessor
from .deim_criterion import DEIMCriterion
from .simple_encoder import SimpleEncoder
from .cross_fused_encoder import CrossFusedEncoder
