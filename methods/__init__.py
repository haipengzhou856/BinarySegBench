from .segformer import build_segformer
from .DPT_custom import DPT

def GetModel(config):
    model_name = config.MODEL.NAME
    if model_name == 'Segformer':
        return build_segformer(config)
    elif model_name == 'DPT':
        # return build_DPT(config)
        return DPT(encoder_size="base",
                   backbone="DINOV2",
                   nclass=1)