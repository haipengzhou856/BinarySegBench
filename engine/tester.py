import os
import torch
from tqdm import tqdm
from utils.utils import Tensor2PIL, check_dir
import time

def eval_function(cfg, model, test_loader, result_path, epoch):
    torch.cuda.synchronize()
    time_start = time.time()
    for _, batch in enumerate(tqdm(test_loader)):
        img, label, gt_path = batch["image"], batch["label"], batch["mask_path"]
        ori_h, ori_w = batch["ori_h"][0], batch["ori_w"][0],

        # this is for hugging face testing
        with torch.no_grad():
            # huggingface style output
            prediction = model(img)
            logits = prediction["logits"]

            resized_pred = torch.nn.functional.interpolate(
                logits, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
            for pred, path in zip(resized_pred, gt_path):
                pred = (pred > 0.5).to(float) # threshold of probability is 0.5
                img = Tensor2PIL(pred)
                folder_path = os.path.join(result_path, "ckpt_epoch{epoch}".format(epoch=epoch))
                check_dir(folder_path)
                cur_path = os.path.join(folder_path, path.split("/")[-1:][0]) # path.split("/")[-1:][0]: image name
                img.save(cur_path)

    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    return time_sum


