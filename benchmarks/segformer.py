from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional

def build_segformer(model_name,num_class):
    if "segformer" in model_name:
        model = SegformerForSemanticSegmentation.from_pretrained(
        "/home/haipeng/Code/hgf_pretrain/nvidia/segformer-b3-finetuned-ade-512-512",
        ignore_mismatched_sizes=True,
        num_labels=num_class)
    else:
        raise ValueError("NO MODEL IMPLEMENTED")
    return model


