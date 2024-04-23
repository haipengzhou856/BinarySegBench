# stolen from torchvision.models.segmentation.deeplabv3_resnet101
# Thanks for the torch team

import torchvision
import torch
from torch import nn
from typing import List, Optional
from transformers.modeling_outputs import SemanticSegmenterOutput

'''class Deeplabv3(nn.Module):
    def __init__(self,
                 PretrainedModel
                 ):
        super().__init__()
        self.Model = PretrainedModel

    def forward(self, images):
        outputs = self.Model(images)
        logits = outputs["out"]

        # keep the same output style with huggingface transformers
        return SemanticSegmenterOutput(loss=None, logits=logits)'''


class Deeplabv3(nn.Module):
    def __init__(self,
                 PretrainedModel, num_class
                 ):
        super().__init__()
        self.Model = PretrainedModel
        self.Model.classifier[4] = nn.Conv2d(256, num_class, kernel_size=(1, 1), stride=(1, 1))
        nn.init.kaiming_normal_(self.Model.classifier[4].weight, mode='fan_in')

    def forward(self, images):
        outputs = self.Model(images)
        logits = outputs["out"]

        # keep the same output style with huggingface transformers
        return SemanticSegmenterOutput(loss=None, logits=logits)


def build_deeplabv3(model_name, num_class):
    if "deeplabv3" in model_name:
        # PretrainedModel = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=num_class, progress=True,pretrained_backbone=True)
        PretrainedModel = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=21, progress=True,
                                                                              pretrained=True)
    else:
        raise ValueError("NO MODEL IMPLEMENTED")
    return Deeplabv3(PretrainedModel, num_class)


if __name__ == '__main__':
    PretrainedModel = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=21, progress=True,
                                                                          pretrained=True)
