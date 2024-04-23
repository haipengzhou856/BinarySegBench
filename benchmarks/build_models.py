import torch
from .deeplab import build_deeplabv3
from .segformer import build_segformer
from .UperNet import build_upernet

def build_benchmarks(model_name, num_class):
    _available_models = {"deeplabv3": build_deeplabv3("deeplabv3", num_class),
                         "segformer": build_segformer("segformer", num_class),
                         "upernet-swin-s":build_upernet("upernet-swin-s", num_class)}
    assert model_name in _available_models.keys()
    return _available_models[model_name]
