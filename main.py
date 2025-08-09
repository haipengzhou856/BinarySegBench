import os

# Required for 4000 Series, comment out for other models
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# import swanlab
import argparse
from configs import get_cfg_default
import accelerate
from accelerate.utils import set_seed
from utils.utils import (
    SetDirectory,
    cal_param,
    load_module_from_path,
    setup_main_process_logger,
    setup_main_process_writer
)
import torch.distributed as dist
from data.Loader import GetIMGLoader
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
from methods import GetModel
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default='configs/datasets/DIS5K.yaml',
                        help='config of the dataset')
    parser.add_argument('--model_config', type=str, default='configs/models/Segformer.py',
                        help='config of the model')
    parser.add_argument('--exp_name', type=str, default='debug_mode',
                        help='name your experiment')
    parser.add_argument('--custom_name', type=str, default='',
                        help='name your ablation')
    parser.add_argument('--model_name', type=str, default='Segformer',
                        help='basic name')
    parser.add_argument('--data_input', type=str, help='input path')
    parser.add_argument('--data_output', type=str, help='output path')

    args = parser.parse_args()

    # Freeze configuration
    cfg = get_cfg_default()
    cfg.EXP_NAME = args.exp_name
    cfg.MODEL.NAME = args.model_name
    cfg.merge_from_file(args.dataset_config)

    module = load_module_from_path(args.model_config)
    assert hasattr(module, 'extend_cfg'), "Check the model config file, extend_cfg function is not found"
    module.extend_cfg(cfg)

    cfg.DATASET.ROOT = args.data_input
    cfg.OUTPUT_DIR = args.data_output
    # cfg.DATASET.ROOT = os.path.join(os.environ["DATASET_PATH"], cfg.DATASET.ROOT)

    cfg.freeze()

    #############################################################################
    ################################  Set miscellaneous #########################
    #############################################################################
    # Tested on several machines
    # If you encounter program hang/stuck or distributed training related issues
    # comment out dist.init_process_group(backend="gloo")
    # Required for A6000 & HPC
    # but 4090 works well without it
    # Presumed to be due to environment differences
    # dist.init_process_group(backend="gloo")

    # Set all parameters
    out_dir, ckpt_path, result_path, tb_path, log_path = SetDirectory(
        cfg, custom_name=args.custom_name
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=2
    )
    # accelerator = accelerate.Accelerator()

    device = accelerator.device
    logger = setup_main_process_logger(cfg, accelerator, log_path)
    writer = setup_main_process_writer(cfg, accelerator, tb_path)

    set_seed(cfg.SEED)

    #############################################################################
    ################################  Log basics ################################
    #############################################################################
    logger.info("----------------------NEW RUN----------------------------")
    logger.info("-------------------Basic Setting-------------------------")
    logger.info("---output in: {dir}---".format(dir=out_dir))
    logger.info("Img_size: {}".format(cfg.INPUT.SIZE))
    logger.info("BATCH_SIZE: {}".format(cfg.DATALOADER.BATCH_SIZE))
    logger.info("scheduler: {}".format(cfg.SOLVER.SCHEDULER))
    logger.info("lr: {}".format(cfg.SOLVER.LR))
    logger.info("opim: {}".format(cfg.SOLVER.OPTIMIZER))
    logger.info(
        "Using {num_gpu} GPU for training, {mix_pix} used.".format(
            num_gpu=accelerator.num_processes,
            mix_pix=accelerator.mixed_precision
        )
    )

    #############################################################################
    ################################  Get dataloader ############################
    #############################################################################
    # dataset_name = cfg.DATASET.NAME
    # logger.info("-----------------EXP on {dataset_name}----------------".format(dataset_name=dataset_name))

    dataloaders = {}
    dataloaders['train'] = GetIMGLoader(cfg, is_train=True)
    dataloaders['test'] = GetIMGLoader(cfg, is_train=False)
    # dataloaders['train'] = GetRefLoader(cfg, is_train=True)
    # dataloaders['test'] = GetRefLoader(cfg, is_train=False)

    logger.info("-----------------Finish dataloader----------------")

    #############################################################################
    ################################  Create model ##############################
    #############################################################################
    # Define the model

    model = GetModel(cfg)


    logger.info("--------------Finish creating model----------------")

    #############################################################################
    ################################  Train scheme ##############################
    #############################################################################
    from engine.trainer import training_func

    training_func(cfg, accelerator, model, dataloaders, logger, writer, ckpt_path, result_path)

    logger.info("-----------------End RUN!----------------")