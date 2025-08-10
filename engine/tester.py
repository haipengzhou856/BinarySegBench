import os
import torch
from tqdm import tqdm
from utils.utils import Tensor2PIL, check_dir
import time
from PIL import Image
import numpy as np
import torch.nn.functional as F
from utils.misc import masks_to_boxes, masks_sample_points, visualize_with_box_and_points
from utils.metrix_matte import SAD, MSE, MAD, CONN, Grad


def eval_func(cfg, model, test_loader, result_path, epoch, accelerator):
    metrics = {
        "sad": SAD(),
        "mse": MSE(),
        # "mad": MAD(),
        # "conn": CONN(),
        # "grad": Grad()
    }



    inference_time_sum = 0.0
    total_samples = 0

    model.eval()

    for _, batch in enumerate(tqdm(test_loader, desc="---------Evaluating:")):
        img, labels, gt_idx, ori_hw = batch["image"], batch["label"], batch["mask_idx"], batch["ori_hw"]
        batch_size = img.shape[0]
        total_samples += batch_size

        box = masks_to_boxes(labels[:, 0, ])
        box = box.to(labels.device)

        points = masks_sample_points(labels[:, 0, ], k=3)
        points_label_pos = torch.ones(points.shape[:2])
        points = points.to(labels.device)
        points_label_pos = points_label_pos.to(labels.device)

        with torch.no_grad():
            start_time = time.time()
            prediction = model(img)
            end_time = time.time()
            inference_time_sum += end_time - start_time

            logits = prediction["logits"]
            logits = torch.nn.functional.interpolate(logits, labels.shape[2:], mode="bilinear", align_corners=False)
            # logits = torch.sigmoid(logits)

            all_logits, all_labels, all_idx = accelerator.gather_for_metrics((logits, labels, gt_idx))
            gt_path = [test_loader.dataset.idx_to_mask_path[idx] for idx in all_idx.cpu().detach().numpy()]

            if accelerator.is_main_process:
                pred_np = all_logits.cpu().detach().numpy()
                pred_np = pred_np.squeeze(1)
                gt_alpha_np = all_labels[:, 0].cpu().detach().numpy()

                for i in range(pred_np.shape[0]):
                    pred = pred_np[i]
                    gt = gt_alpha_np[i]

                    for name, metric in metrics.items():
                        val = metric.compute_metric(pred, gt)
                        metric.update(val, 1)

                    img_name = os.path.basename(gt_path[i]).split('.')[0]
                    pred_img = Tensor2PIL(torch.from_numpy(pred).unsqueeze(0))
                    pred_img.save(os.path.join(result_path, f"{img_name}_pred.png"))

    return {
        "inference_time": inference_time_sum / total_samples if total_samples > 0 else 0,
        "metrics": {name: metric.avg for name, metric in metrics.items()}
    }
