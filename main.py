import yaml
from utils.utils import *
import accelerate
from accelerate import DistributedDataParallelKwargs
import time
from torch.utils.tensorboard import SummaryWriter
from accelerate.utils import set_seed
import argparse
from dataset.data_processor import GetDataLoader
from engine.trainer import training_func

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--manual_flag', type=str, help='add flag to distinguish exp')
    parser.add_argument('--exp_mode', type=str, default="benchmark",
                        help='your model or benchmark model, only `own` or `benchmark` is used ')
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:  # change the config here
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    set_seed(config.SEED)
    ### I've tested on serveral machines
    ### If you meet programme hang/stuck or distributed training related problem
    ### comment the dist.init_process_group(backend="gloo")
    ### A6000 & HPC need it
    ### but 4090 works very well without it
    import torch.distributed as dist
    dist.init_process_group(backend="gloo")

    # check out_dir
    # manual_flag = time.strftime("%m%d%H%M", time.localtime())  # add timestamp to distinguish
    manual_flag = args.manual_flag
    out_dir = config.OUTPUT.HOME + config.OUTPUT.MODEL_NAME + "/" + config.OUTPUT.EXP_NAME + "/" + manual_flag

    # HOME/MODEL_NAME/EXP_NAME/manual_flag
    tb_path = check_dir(out_dir + config.OUTPUT.TB)  # tensorboard
    ckpt_path = check_dir(out_dir + config.OUTPUT.CKPT)  # checkpoint
    log_path = check_dir(out_dir + config.OUTPUT.LOG)  # logs
    result_path = check_dir(out_dir + config.OUTPUT.RESULT)  # store the test results
    # copy the folder name for save results, avoiding process preemption in acceleration when mkdir

    # if some layer not used, use it to avoid the errors
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    accelerator = accelerate.Accelerator()

    writer = SummaryWriter(tb_path)
    # logger init
    logger = setup_logger(config.OUTPUT.MODEL_NAME,
                          log_path,
                          accelerator.process_index,
                          "log.txt")

    # logs
    logger.info("----------------------NEW RUN----------------------------")
    logger.info("-------------------Basic Setting-------------------------")
    logger.info("---work place in: {dir}---".format(dir=out_dir))
    logger.info("Img_size: {}".format(config.DATASET.IMG_SIZE))
    logger.info("BATCH_SIZE: {}".format(config.DATASET.BATCH_SIZE))
    logger.info("scheduler: {}".format(config.SOLVER.LINEAR_SCHEDULE))  # need manually set
    logger.info("lr: {}".format(config.SOLVER.OPTIM.LR))
    logger.info("optim: {}".format(config.SOLVER.OPTIM.NAME))
    logger.info(
        "--------------------USE {model_name}-----------------------".format(model_name=config.OUTPUT.MODEL_NAME))
    logger.info(
        "Using {num_gpu} GPU for training, {mix_pix} mix_precision used.".format(num_gpu=accelerator.num_processes,
                                                                                 mix_pix=accelerator.mixed_precision))

    # import the models here
    model_name = config.OUTPUT.MODEL_NAME
    model_size = config.OUTPUT.EXP_NAME # TODO, e.g., swin-b or -s
    num_class = config.DATASET.NUM_CLASSES

    ## YOU NEED MANUALLY IMPORT THE MODEL HERE
    ## YOU NEED MANUALLY IMPORT THE MODEL HERE
    ## YOU NEED MANUALLY IMPORT THE MODEL HERE
    exp_mode = args.exp_mode
    if exp_mode == "benchmark":
        from benchmarks.build_models import build_benchmarks
        model = build_benchmarks(model_name, num_class)
    else:
        # from models import your_model
        model = None  # YOURS TODO

    train_loader = GetDataLoader(config, is_train=True)
    test_loader = GetDataLoader(config, is_train=False)
    dataloaders = {}
    dataloaders["train"] = train_loader
    dataloaders["test"] = test_loader

    logger.info("-----------------Finish dataloader----------------")

    training_func(config, accelerator, model, logger, writer, ckpt_path, result_path, dataloaders)
    # TODO
    # eval()
    logger.info("----------------------END RUN----------------------------")
