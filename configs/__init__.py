################ steal this setting from Dassl and Detectron2 ################
######################### big thanks to the authors ##########################

from .defaults import _C as cfg_default
from yacs.config import CfgNode as CN
import yaml

def get_cfg_default():
    return cfg_default.clone()


