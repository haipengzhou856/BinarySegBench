from transformers import UperNetForSemanticSegmentation
import torch
from torch import nn
from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional

def build_upernet(model_name,num_class):
    if "upernet-swin-b" in model_name:
        model = UperNetForSemanticSegmentation.from_pretrained(
        "/home/haipeng/Code/hgf_pretrain/openmmlab/upernet-swin-base",
        ignore_mismatched_sizes=True,
        num_labels=num_class)
    elif "upernet-swin-s" in model_name:
        model = UperNetForSemanticSegmentation.from_pretrained(
        "/home/haipeng/Code/hgf_pretrain/openmmlab/upernet-swin-small",
        ignore_mismatched_sizes=True,
        num_labels=num_class)
    else:
        raise ValueError("NO MODEL IMPLEMENTED")
    return model


if __name__ == '__main__':
    model = build_upernet("upernet-swin-s",12)
    img = torch.randn(2,3,768,768)
    output = model(img)
    print("qdq")