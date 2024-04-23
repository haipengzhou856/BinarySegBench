from tqdm import tqdm
from utils.utils import *
from accelerate.utils import set_seed
from solver.loss_func import bce_loss, hinge_loss
from solver.build_solver import build_optimizer, build_scheduler
from .tester import eval_function
from tqdm import tqdm
import numpy as np
import os
from PIL import Image

## custom dataset evaluation metrix funciton
from utils.metrix_BIGSHA import computeIOU_MAE_BER, computeBER_mth


def training_func(cfg, accelerator, model, logger, writer, ckpt_path, result_path, dataloaders):
    set_seed(cfg.SEED)
    train_loader = dataloaders["train"]
    test_loader = dataloaders["test"]

    # hyps
    total_epoch = cfg.SOLVER.EPOCH
    store_epoch = cfg.OUTPUT.STORE_EPOCH
    device = accelerator.device
    resume_path = cfg.MODEL.RESUME_PATH
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    gt_path = cfg.DATASET.DATA_ROOT + "/test/mask"
    # setup training modules
    model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, test_loader)

    logger.info("----------------Starting training------------------")
    logger.info(f"--------------Total  {total_epoch} Epochs--------------")

    accelerator.print(resume_path)
    if resume_path != "":  # if resume
        logger.info(f"Resumed from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)
        path = os.path.basename(resume_path)
        starting_epoch = int(path.replace("ckpt_epoch", "")) + 1
    else:
        starting_epoch = 1

    overall_step = 0
    best_epoch = 0
    for epoch in range(starting_epoch, total_epoch + 1):
        for idx, batch in enumerate(tqdm(train_loader, disable=not accelerator.is_local_main_process)):
            model.train()
            imgs, labels, = batch["image"], batch["label"]
            ################# training detail ###################
            #### model gives logits, add loss function here #####
            ############### or modify it customly ###############
            outputs = model(imgs)
            loss1 = bce_loss(outputs.logits, labels)
            loss2 = hinge_loss(outputs.logits, labels)
            loss = loss1 + 0.2 * loss2
            accelerator.backward(loss)
            ################# update optimizer #################
            optimizer.step()
            if cfg.SOLVER.CLR_SCEDULE.IS_USE != 1:
                scheduler.step()
            optimizer.zero_grad()
            overall_step += 1
            ###################### monitor #####################
            writer.add_scalar("loss", loss, overall_step)
            writer.add_scalar("lr:", optimizer.state_dict()['param_groups'][0]['lr'], overall_step)
            writer.add_image("pred_{ii}".format(ii=0), (outputs.logits[0]).to(float), overall_step)
            writer.add_image("gt_{ii}".format(ii=0), labels[0], overall_step)
            writer.add_image("img_{ii}".format(ii=0), reverse_normalize(imgs[0]), overall_step)
            if overall_step % 10 == 0:
                if accelerator.is_main_process:
                    logger.info("Current step:{step}, loss:{loss}, epoch:{epoch}, lr:{lr}".format(
                        step=overall_step, loss=loss, epoch=epoch,
                        lr=optimizer.state_dict()['param_groups'][0]['lr']))

        # if cfg.SOLVER.CLR_SCEDULE.IS_USE == 1:
        #    scheduler.step(epoch)
        ######################### store and eval #####################

        # origin is bellow
        if epoch % store_epoch == 0:
            logger.info(
                "----------------Save ckpt_epoch{epoch}------------------".format(epoch=epoch))
            restore_path = os.path.join(ckpt_path, "ckpt_epoch{epoch}".format(epoch=epoch))
            accelerator.save_state(restore_path)

            logger.info(
                "----------Starting Testing, now is step:{step} epoch:{epoch}-----------".format(
                    step=overall_step,
                    epoch=epoch))
            model.eval()
            infer_time = eval_function(cfg, model, test_loader, result_path, epoch)
            # infer_time will not be used here to compute the FPS
            # using the eval_function via inference.py independently for fair comparisons
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                logger.info(
                    "--------------------RECORD MATRICX----------------------")
                metrix = computeIOU_MAE_BER(gt_path=gt_path,
                                            pred_path=os.path.join(result_path,
                                                                   "ckpt_epoch{epoch}".format(epoch=epoch)))
                logger.info(
                    "IoU:{IoU},MAE:{mae},BER:{BER},SBER:{SB},NBER:{NB}".format(IoU=metrix["IoU"],
                                                                               mae=metrix["MAE"],
                                                                               BER=metrix["BER"],
                                                                               SB=metrix["S-BER"],
                                                                               NB=metrix["N-BER"]))
                writer.add_scalar("IoU", metrix["IoU"], epoch)
                writer.add_scalar("BER", metrix["BER"], epoch)
                writer.add_scalar("MAE", metrix["MAE"], epoch)
