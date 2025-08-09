import torch
from torch import nn
from sam2.build_sam import build_sam2
from typing import Any, Dict, List, Tuple
from PIL import Image
import matplotlib.pylab as plt
import numpy as np
from transformers.modeling_outputs import SemanticSegmenterOutput


class CustomSAM2Img(nn.Module):
    # static version sam2
    def __init__(self, config=None):
        super().__init__()
        checkpoint = "/home/haipeng/Pretrained/sam2/sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        self.sam = build_sam2(model_cfg, checkpoint)
        del self.sam.memory_encoder
        del self.sam.memory_attention


    def get_img_embedding(self, img):
        backbone_out = self.sam.forward_image(img)
        backbone_out, vision_feats, vision_pos_embeds, feat_sizes=self.sam._prepare_backbone_features(backbone_out)
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def get_box_embedding(self, input_box_tensor):
        return

    def get_point_embedding(self, input_point_tensor):
        return

    def get_mask_embedding(self, input_mask_tensor):
        return

    @torch.no_grad()
    def forward(self,batched_input: List[Dict[str, Any]],)-> List[Dict[str, torch.Tensor]]:
        ############################### image embedding#####################################

        return None

if __name__ == '__main__':
    checkpoint = "/home/haipeng/Pretrained/sam2/sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    model = build_sam2(model_cfg, checkpoint)
    model.cuda()
    a = torch.randn(4,3,1024,1024).cuda()
    img_emd = model.get_img_embedding(a)

