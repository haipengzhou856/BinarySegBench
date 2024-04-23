import torch
from timm.scheduler import CosineLRScheduler


def build_optimizer(cfg, model):
    lr = cfg.SOLVER.OPTIM.LR
    momentum = cfg.SOLVER.OPTIM.MOMENTUM
    decay = cfg.SOLVER.OPTIM.DECAY

    if cfg.SOLVER.OPTIM.NAME == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr, momentum=momentum, weight_decay=decay)
    elif cfg.SOLVER.OPTIM.NAME == "AdamW":
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=lr, weight_decay=decay)
    elif cfg.SOLVER.OPTIM.NAME == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=lr, weight_decay=decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIM.NAME}")
    return optimizer


def build_scheduler(cfg, optimizer):
    total_epoch = cfg.SOLVER.EPOCH

    if cfg.SOLVER.LINEAR_SCHEDULE.IS_USE == 1:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.LINEAR_SCHEDULE.MILESTONGS,
                                                         gamma=cfg.SOLVER.LINEAR_SCHEDULE.GAMMA, last_epoch=-1)
    elif cfg.SOLVER.CA_SCHEDULE.IS_USE == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.CA_SCHEDULE.TMAX,
                                                               last_epoch=-1)
    elif cfg.SOLVER.CLR_SCEDULE.IS_USE == 1:
        scheduler = CosineLRScheduler(optimizer, warmup_lr_init=cfg.SOLVER.CLR_SCEDULE.WARM_LR, t_initial=total_epoch,
                                      cycle_decay=cfg.SOLVER.CLR_SCEDULE.CYCLE_DECAY,
                                      warmup_t=cfg.SOLVER.CLR_SCEDULE.WARM_EPOCH)
    else:
        raise ValueError(f"Unsupported lr_schduler: {cfg.SOLVER.LR_SCHEDULE}")
    return scheduler