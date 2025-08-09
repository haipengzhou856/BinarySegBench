from transformers.modeling_outputs import SemanticSegmenterOutput
from torch import nn
from typing import Optional
import torch
import torch.nn.functional as F
from .dpt_head import DPTHead
###### backbones
from transformers import AutoModel, Dinov2Backbone


def DINOv2(model_name):
    model_zoo = {
        "small": torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'),
        "base": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14'),
        "large": torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14'),
    }
    return model_zoo[model_name]



class DPT(nn.Module):
    def __init__(
            self,
            encoder_size='base',
            backbone = "DINOV2",
            nclass=21,
            use_bn=False,
    ):
        super(DPT, self).__init__()
        model_configs = {
            'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39]
        }

        self.encoder_size = encoder_size
        self.backbone_name = backbone
        if backbone == "DINOV2":
            self.backbone = DINOv2(model_name=encoder_size)



        self.backbone.patch_size=self.patch_size = 14
        self.embed_dim = self.backbone.embed_dim
        self.head = DPTHead(nclass, self.embed_dim, model_configs[encoder_size]["features"], use_bn, out_channels=model_configs[encoder_size]["out_channels"])
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)


    def forward(self, x, comp_drop=False):
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size


        if self.backbone_name == "DINOV2":
            features = self.backbone.get_intermediate_layers(
                x, self.intermediate_layer_idx[self.encoder_size]
            )

        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * self.patch_size, patch_w * self.patch_size), mode='bilinear', align_corners=True)
        out = F.sigmoid(out)
        return SemanticSegmenterOutput(logits=out)




