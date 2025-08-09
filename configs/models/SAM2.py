from yacs.config import CfgNode as CN

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    cfg.MODEL.SAM2 = CN()
    cfg.MODEL.SAM2.CONFIG = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    cfg.MODEL.SAM2.PRETRAINED_WEIGHT = '/home/haipeng/Pretrained/sam2/sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt'



