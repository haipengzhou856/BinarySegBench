from tqdm import tqdm
from timm.utils.metrics import AverageMeter, accuracy
from torch.cuda.amp import autocast
import os
from accelerate.utils import set_seed
from solver.build_solver import build_optimizer, build_scheduler
from solver.loss_func import bce_loss, hinge_loss
from solver.matting_loss import laplacian_loss, regression_loss,composition_loss,gradient_loss
from torch.nn import functional as F
from .tester import eval_func
from utils.utils import *
from utils.metrix_v1 import calculate_metrics
from utils.metrix_matte import matte_evaluate_metrics
import torch.distributed as dist
import time
from utils.misc import masks_to_boxes,masks_sample_points, visualize_with_box_and_points

GT_PATH = "/home/haipeng/Data/DIS5K/test/gt"



def training_func(cfg, accelerator, model, dataloaders, logger, writer, ckpt_path, result_path):
    set_seed(cfg.SEED)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    train_loader, test_loader = dataloaders['train'], dataloaders['test']

    total_epoch = cfg.SOLVER.EPOCH
    store_epoch = cfg.SOLVER.STORE_EPOCH
    device = accelerator.device
    resume_path = cfg.RESUME_DIR  # ONLY USED FOR EMERGENCY SITUATION, NEED MANUALLY CHANGE CONFIG

    model, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, test_loader)
    logger.info("----------------Starting training------------------")
    logger.info(f"--------------Total {total_epoch} Epochs--------------")
    accelerator.print(resume_path)
    if resume_path != "":  # if resume
        logger.info(f"Resumed from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)
        path = os.path.basename(resume_path)
        starting_epoch = int(path.replace("ckpt_epoch", ""))
    else:
        starting_epoch = 0

    overall_step = 0
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, total_epoch + 1):
        epoch = epoch + starting_epoch
        model.train()
        for _, batch in enumerate(tqdm(train_loader)):
            images, labels = batch["image"], batch['label']
            box = masks_to_boxes(labels[:, 0, ])
            box = box.to(labels.device)

            ### points, now only for positive point
            points = masks_sample_points(labels[:, 0, ], k=3)
            points_label_pos = torch.ones(points.shape[:2])
            points = points.to(labels.device)
            points_label_pos = points_label_pos.to(labels.device)
            points_pair = (points,points_label_pos)

            # labels = labels.repeat(1, 3, 1, 1)
            # outputs = model(images,labels)
            # loss = outputs

            ### baseline
            #outputs = model(images,box,points_pair)
            outputs = model(images)
            #loss = bce_loss(outputs.logits, labels)  # +0.5*hinge_loss(outputs.logits, labels)     # seg loss
            loss = regression_loss(outputs.logits, labels) + laplacian_loss(outputs.logits, labels)+ \
                   composition_loss(outputs.logits, labels,images)+gradient_loss(outputs.logits, labels)
            # matting loss


            accelerator.backward(loss)
            ################# update optimizer #################
            optimizer.step()
            optimizer.zero_grad()
            overall_step += 1
            ###################### monitor #####################
            writer.add_scalar("loss", loss, overall_step)
            writer.add_scalar("lr:", optimizer.state_dict()['param_groups'][0]['lr'], overall_step)
            if overall_step % 10 == 0:
                logger.info("Current step:{step}, loss:{loss}, epoch:{epoch}, lr:{lr}".format(
                    step=overall_step, loss=loss, epoch=epoch, lr=optimizer.state_dict()['param_groups'][0]['lr']))
                writer.add_image("pred_{ii}".format(ii=0), outputs.logits[0], overall_step)
                writer.add_image("img_{ii}".format(ii=0), reverse_normalize(images[0]), overall_step)
                vis_label = visualize_with_box_and_points(labels,box,points,idx=0)
                writer.add_image("gt_{ii}".format(ii=0), vis_label, overall_step)


            if overall_step % 5 == 0:
                if accelerator.is_main_process:
                    logger.info("Current step:{step}, loss:{loss}, epoch:{epoch}, lr:{lr}".format(
                        step=overall_step, loss=loss, epoch=epoch,
                        lr=optimizer.state_dict()['param_groups'][0]['lr']))
        scheduler.step(epoch)
        if epoch % store_epoch == 0:
            model.eval()
            logger.info(f"----------------Save ckpt_epoch{epoch}------------------")
            restore_ckpt_path = os.path.join(ckpt_path, f"ckpt_epoch{epoch}")
            max_wait = 3
            current_wait = 0
            accelerator.save_state(restore_ckpt_path)  # TODO: bugs when using Poly scheduler and cant to save
            infer_time = eval_func(cfg, model, test_loader, result_path, epoch)
            accelerator.wait_for_everyone()
            # dist.barrier()

            '''if accelerator.is_main_process:
                logger.info(
                    "--------------------RECORD MATRICX----------------------")
                logger.info(f"----------------Raw Infer Time: {infer_time}------------------")

                # TODO
                GT_PATH = cfg.DATASET.ROOT + "/test/gt"
                accelerator.wait_for_everyone()
                metrix = matte_evaluate_metrics(directory_gt=GT_PATH,
                                             directory_mask=os.path.join(result_path,
                                                                         "ckpt_epoch{epoch}".format(epoch=epoch)),
                                                num_threads=64)
                writer.add_scalar("SAD", metrix['SAD'], epoch)
                writer.add_scalar("MSE", metrix['MSE'], epoch)
                writer.add_scalar("GRAD", metrix['GRAD'], epoch)
                writer.add_scalar("CONN", metrix['CONN'], epoch)
                writer.add_scalar("MAD", metrix['MAD'], epoch)'''



