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
    cfg.MODEL.Segformer = CN()
    cfg.MODEL.Segformer.PRETRAINED_WEIGHT = '/home/haipeng/Pretrained/nvidia/segformer-b5-finetuned-ade-640-640'



