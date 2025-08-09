import os
import torch
from tqdm import tqdm
from utils.utils import Tensor2PIL, check_dir
import time
from PIL import Image
import numpy as np
import torch.nn.functional as F
from utils.misc import masks_to_boxes,masks_sample_points, visualize_with_box_and_points

def eval_func(cfg, model, test_loader, result_path, epoch):
    inference_time_sum = 0
    for _, batch in enumerate(tqdm(test_loader)):
        img, labels, gt_path, ori_hw = batch["image"], batch["label"], batch["mask_path"], batch["ori_hw"]
        box = masks_to_boxes(labels[:, 0, ])
        box = box.to(labels.device)

        ### points, now only for positive point
        points = masks_sample_points(labels[:, 0, ], k=3)
        points_label_pos = torch.ones(points.shape[:2])
        points = points.to(labels.device)
        points_label_pos = points_label_pos.to(labels.device)
        points_pair = (points, points_label_pos)

        # z_img, z_label = batch['z_image'], batch['z_mask']
        # this is for hugging face testing
        with torch.no_grad():
            if cfg.MODEL.NAME != "SAM2":
                torch.cuda.synchronize()
                start_time = time.time()
                #prediction = model(img,box,points_pair)
                prediction = model(img)
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time_sum += end_time - start_time

                logits = prediction["logits"]
                for i in range(len(logits)):
                    resized_pred = torch.nn.functional.interpolate(
                        logits[i].unsqueeze(0), size=(int(ori_hw[0][i]),int(ori_hw[1][i])), mode="bilinear",align_corners=False)
                    pred = resized_pred.squeeze(0).to(float)

                    # pred = F.sigmoid(pred)
                    # pred = (pred > 0.5).to(float)
                    img = Tensor2PIL(pred)
                    img = img.convert("L")
                    folder_path = os.path.join(result_path, "ckpt_epoch{epoch}".format(epoch=epoch))
                    check_dir(folder_path)
                    cur_path = os.path.join(folder_path, gt_path[i].split("/")[-1:][0])  # gt_path[i].split("/")[-1:][0]: image name
                    img.save(cur_path)
            else:
                bbox = model.transforms.transform_boxes(bbox, normalize=True, orig_hw=(ori_hw[0], ori_hw[1]))
                torch.cuda.synchronize()
                start_time = time.time()
                prediction = model(img, bbox)
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time_sum += end_time - start_time

                logits = prediction["logits"]
                for i in range(len(logits)):
                    resized_pred = torch.nn.functional.interpolate(
                        logits[i].unsqueeze(0), size=ori_hw, mode="bilinear",
                        align_corners=False)
                    pred = resized_pred > 0.0
                    pred = pred.squeeze(0).float().detach().cpu().numpy()
                    pred = pred.squeeze(0)
                    pred = pred * 255
                    img = Image.fromarray(pred).convert("L")
                    folder_path = os.path.join(result_path, "ckpt_epoch{epoch}".format(epoch=epoch))
                    check_dir(folder_path)
                    cur_path = os.path.join(folder_path,
                                            gt_path[i].split("/")[-1:][0])  # gt_path[i].split("/")[-1:][0]: image name
                    img.save(cur_path)

    return inference_time_sum