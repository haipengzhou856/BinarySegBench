import yaml
from utils.utils import *
import accelerate
from accelerate import DistributedDataParallelKwargs
import time
from torch.utils.tensorboard import SummaryWriter
from accelerate.utils import set_seed
import argparse
from utils.metrix_BIGSHA import computeALL
from dataset.data_processor import GetDataLoader
from engine.tester import eval_function



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--manual_flag', type=str, help='add flag to distinguish')
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:  # change the config here
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    set_seed(config.SEED)

    # check out_dir
    # manual_flag = time.strftime("%m%d%H%M", time.localtime())  # add timestamp to distinguish
    manual_flag = args.manual_flag

    out_dir = config.OUTPUT.HOME + config.OUTPUT.MODEL_NAME + "/" + config.OUTPUT.EXP_NAME + "/" + manual_flag

    # HOME/MODEL_NAME/EXP_NAME/manual_flag
    tb_path = check_dir(out_dir + config.OUTPUT.TB)  # tensorboard
    log_path = check_dir(out_dir + config.OUTPUT.LOG)  # logs
    result_path = check_dir(out_dir + config.OUTPUT.RESULT)  # store the test results
    # copy the folder name for save results, avoiding process preemption in acceleration when mkdir
    # if some layer not used, use it to avoid the warning
    accelerator = accelerate.Accelerator()

    # if accelerator.num_processes>1:
    #    import torch.utils.data.distributed
    #    import torch.distributed as dist
    #    dist.init_process_group(backend="gloo")

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
    logger.info("opim: {}".format(config.SOLVER.OPTIM.NAME))

    logger.info(
        "--------------------USE {model_name}-----------------------".format(model_name=config.OUTPUT.MODEL_NAME))

    logger.info("-------------------INFERENCE STAGE----------------------")
    logger.info("-------------------INFERENCE STAGE----------------------")
    logger.info("-------------------INFERENCE STAGE----------------------")

    # import the models here
    model_name = config.OUTPUT.MODEL_NAME
    num_class = config.DATASET.NUM_CLASSES
    model_size = config.OUTPUT.EXP_NAME
    torch.cuda.reset_max_memory_allocated()

    from benchmarks.build_models import build_benchmarks
    model = build_benchmarks(model_name, num_class)

    test_loader = GetDataLoader(config, is_train=False)
    logger.info("-----------------Finish dataloader----------------")

    model, test_loader = accelerator.prepare(
        model, test_loader)

    weight_path = config.MODEL.CKP_PATH
    accelerator.print(weight_path)
    logger.info(f"Load weight from checkpoint: {weight_path}")
    accelerator.load_state(weight_path)
    path = os.path.basename(weight_path)
    epoch = int(path.replace("ckpt_epoch", ""))

    model.eval()
    infer_time = eval_function(config, model, test_loader, result_path, epoch)
    peak_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # convert into MB
    print("Infer Mem：{:.2f} MB".format(peak_memory_usage))
    fps = 200.0 / infer_time  # we total have 200 images for testing stage
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info(
            "--------------------Inference Time: {time}----------------------".format(time=infer_time))
        logger.info("----------------------FPS:{fps}----------------------------".format(fps=fps))
        logger.info(
            "--------------------RECORD MATRICX----------------------")
        gt_path = config.DATASET.DATA_ROOT + "/test/mask"
        metrix = computeALL(gt_path=gt_path,
                            pred_path=os.path.join(result_path, "ckpt_epoch{epoch}".format(epoch=epoch)))
        logger.info(
            "IoU:{IoU},MAE:{mae},BER:{BER},SBER:{SB},NBER:{NB},F-beta:{Fmeasure}".format(IoU=metrix["IoU"],
                                                                                  mae=metrix["MAE"],
                                                                                  BER=metrix["BER"],
                                                                                  SB=metrix["S-BER"],
                                                                                  NB=metrix["N-BER"],
                                                                                  Fmeasure=metrix["Fmeasure"]))
    logger.info("----------------------END RUN----------------------------")
