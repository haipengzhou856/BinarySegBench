## stolen from CascadePSP (https://github.com/hkchengrex/CascadePSP/blob/master/eval_post.py)
## thx for the open-source

import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def average(self):
        return self.sum / (self.count + 1e-8)


def get_disk_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))


def compute_boundary_acc(gt, mask):
    gt = gt.astype(np.uint8)
    mask = mask.astype(np.uint8)

    h, w = gt.shape

    min_radius = 1
    max_radius = (w + h) / 300
    num_steps = 5

    mask_acc = np.zeros(num_steps)

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius - min_radius) / num_steps * i)

        kernel = get_disk_kernel(curr_radius)
        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

        gt_in_bound = gt[boundary_region]
        mask_in_bound = mask[boundary_region]

        num_edge_pixels = boundary_region.sum()
        num_mask_gd_pix = ((gt_in_bound) * (mask_in_bound) + (1 - gt_in_bound) * (1 - mask_in_bound)).sum()

        mask_acc[i] = num_mask_gd_pix / (num_edge_pixels + 1e-8)

    return mask_acc.mean()


def compute_iou(gt, mask):
    intersection = np.logical_and(gt, mask)
    union = np.logical_or(gt, mask)
    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    return iou


def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))

    return loss / 1000


# TODO: may be not correct, need to use matlab code to verify
def compute_gradient_error(pred, gt):
    pred_grad_x = cv2.Sobel(pred, cv2.CV_64F, 1, 0, ksize=3)
    pred_grad_y = cv2.Sobel(pred, cv2.CV_64F, 0, 1, ksize=3)
    gt_grad_x = cv2.Sobel(gt, cv2.CV_64F, 1, 0, ksize=3)
    gt_grad_y = cv2.Sobel(gt, cv2.CV_64F, 0, 1, ksize=3)

    pred_grad = np.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)
    gt_grad = np.sqrt(gt_grad_x ** 2 + gt_grad_y ** 2)

    return np.mean((pred_grad - gt_grad) ** 2)


# TODO: may be not correct, need to use matlab code to verify
def compute_connectivity_error(pred, gt, step=0.1):
    def compute_connectivity(alpha, step):
        h, w = alpha.shape
        alpha = alpha / 255.0
        alpha = np.pad(alpha, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        conn = np.zeros_like(alpha)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                if alpha[i, j] >= step:
                    conn[i, j] = 1
        return conn[1:-1, 1:-1]

    pred_conn = compute_connectivity(pred, step)
    gt_conn = compute_connectivity(gt, step)

    return np.sum(np.abs(pred_conn - gt_conn))


def process_image_metrix1(gt_path, mask_path):
    with Image.open(gt_path) as gt_image, Image.open(mask_path) as mask_im:
        gt = np.array(gt_image.convert('L'))
        mask = np.array(mask_im.convert('L'))

        gt = gt > 128
        mask = mask > 128

        mba = compute_boundary_acc(gt, mask)
        iou = compute_iou(gt, mask)

        return mba, iou


def process_image_metrix2(gt_path, mask_path, trimap_path):
    with Image.open(gt_path) as gt_image, Image.open(mask_path) as mask_im, Image.open(trimap_path) as trimap_im:
        gt = np.array(gt_image.convert('L'))
        mask = np.array(mask_im.convert('L'))
        trimap = np.array(trimap_im.convert('L'))

        sad = compute_sad_loss(mask, gt, trimap)
        mse = compute_mse_loss(mask, gt, trimap)
        conn = compute_connectivity_error(mask, gt)
        grad = compute_gradient_error(mask, gt)
        return sad, mse, conn, grad


import os
from joblib import Parallel, delayed
from tqdm import tqdm


def calculate_metrics(directory_gt, directory_mask, trimap_path=None):
    all_gts = os.listdir(directory_gt)
    num_images = len(all_gts)

    mask_acc_meter = AverageMeter()
    iou_meter = AverageMeter()

    gt_paths = [os.path.join(directory_gt, gt_name) for gt_name in all_gts]
    mask_paths = [os.path.join(directory_mask, gt_name) for gt_name in all_gts]

    results_commone = Parallel(n_jobs=-1)(
        delayed(process_image_metrix1)(gt_path, mask_path)
        for gt_path, mask_path in tqdm(zip(gt_paths, mask_paths), desc='Calculating Segmentation metrics', total=num_images)
    )

    for mask_acc, iou in results_commone:
        mask_acc_meter.update(mask_acc)
        iou_meter.update(iou)

    new_mba = mask_acc_meter.average
    mean_iou = iou_meter.average

    if trimap_path is not None:
        trimap_paths = [os.path.join(trimap_path, gt_name) for gt_name in all_gts]
        results_matte = Parallel(n_jobs=-1)(
            delayed(process_image_metrix2)(gt_path, mask_path, trimap_path)
            for gt_path, mask_path, trimap_path in
            tqdm(zip(gt_paths, mask_paths, trimap_paths), desc='Calculating matte metrics', total=num_images)
        )

        sad_meter = AverageMeter()
        mse_meter = AverageMeter()
        conn_meter = AverageMeter()
        grad_meter = AverageMeter()

        for sad, mse, conn, grad in results_matte:
            sad_meter.update(sad)
            mse_meter.update(mse)
            conn_meter.update(conn)
            grad_meter.update(grad)

        mean_sad = sad_meter.average
        mean_mse = mse_meter.average
        #mean_conn = conn_meter.average
        #mean_grad = grad_meter.average

        return new_mba, mean_iou, mean_sad, mean_mse,# mean_conn, mean_grad

    return new_mba, mean_iou


######### this code is from https://github.com/naver-ai/ZIM/blob/main/eval/metric.py


if __name__ == '__main__':
    '''directory_gt = '/vip_media/haipeng/DIS5K/test/gt'  # Replace with the actual gt directory path
    directory_mask = '/home/haipeng/Matte/ICCV25/output/debug/2025-01-23-19-24-36/ckpt_path/ckpt_epoch4/ckpt_epoch4'  # Replace with the actual mask directory path
    new_mba = calculate_mBA(directory_gt, directory_mask)
    print('New mBA:', new_mba)'''
    # metrics = calculate_Matte('path/to/gt', '/home/haipeng/Matte/ICCV25/output/debug/2025-01-23-19-24-36/ckpt_path/ckpt_epoch4/ckpt_epoch4', grad_filter)
    #mean_ba, mean_iou, mean_sad, mean_mse, mean_conn, mean_grad = calculate_metrics('/vip_media/haipeng/DIS5K/test/gt',
    mean_ba, mean_iou= calculate_metrics('/home/haipeng/Data/DIS5K/test/gt',
                                    '/home/haipeng/Matte/ICCV25/output/inference/2025-02-28-21-58-40/result_path/ckpt_epoch72',
                                                                                    )
    print("mBA:", mean_ba)
    print("mIoU:", mean_iou)
    #print("mSAD:", mean_sad)
    #print("mMSE:", mean_mse)
    #print("mConn:", mean_conn)
    #print("mGrad:", mean_grad)
