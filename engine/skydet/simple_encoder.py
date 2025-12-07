import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from ..core import register


@register()   
class SimpleEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],        
                 feat_strides=[8, 16, 32],            
                 ):
        super().__init__() 

        self.in_channels = in_channels              
        self.feat_strides = feat_strides           
        self.out_channels = in_channels     
        self.out_strides = feat_strides             
 
    def forward(self, feats):   
        assert len(feats) == len(self.in_channels)
        return feats 
