from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional

def build_segformer(config):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path=config.MODEL.Segformer.PRETRAINED_WEIGHT,
        ignore_mismatched_sizes=True,
        num_labels=config.DATASET.NUM_CLASSES)
    return model