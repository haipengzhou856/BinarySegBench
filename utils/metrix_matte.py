import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter


## NOTE: This Metrix only used for observing the training process
##     : For standard comparisons, plz following the matlab code from `Deep Image Matting`

## Some codes from: https://github.com/hmchuong/MaGGIe/blob/main/maggie/utils/metric.py
## Thx for open-source

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SAD(AvgMeter):
    def compute_metric(self, pred, gt):
        pred = pred.astype(np.float32) / 255.0 if pred.max() > 1.0 else pred.astype(np.float32)
        gt = gt.astype(np.float32) / 255.0 if gt.max() > 1.0 else gt.astype(np.float32)

        sad_diff = np.sum(np.abs(pred - gt)) / 1000
        return sad_diff


class MSE(AvgMeter):
    def compute_metric(self, pred, gt):
        pred = pred.astype(np.float32) / 255.0 if pred.max() > 1.0 else pred.astype(np.float32)
        gt = gt.astype(np.float32) / 255.0 if gt.max() > 1.0 else gt.astype(np.float32)

        pixel_count = pred.size
        mse_diff = np.sum((pred - gt) ** 2) / pixel_count
        return mse_diff


class MAD(AvgMeter):
    def compute_metric(self, pred, gt):
        pred = pred.astype(np.float32) / 255.0 if pred.max() > 1.0 else pred.astype(np.float32)
        gt = gt.astype(np.float32) / 255.0 if gt.max() > 1.0 else gt.astype(np.float32)

        pixel_count = pred.size
        mad_diff = np.sum(np.abs(pred - gt)) / pixel_count
        return mad_diff


class Grad(AvgMeter):
    def compute_metric(self, pred, gt, sigma=1.4):
        pred = pred.astype(np.float32) / 255.0 if pred.max() > 1.0 else pred.astype(np.float32)
        gt = gt.astype(np.float32) / 255.0 if gt.max() > 1.0 else gt.astype(np.float32)

        pred_x = gaussian_filter(pred, sigma=sigma, order=(0, 1), mode='constant')
        pred_y = gaussian_filter(pred, sigma=sigma, order=(1, 0), mode='constant')
        gt_x = gaussian_filter(gt, sigma=sigma, order=(0, 1), mode='constant')
        gt_y = gaussian_filter(gt, sigma=sigma, order=(1, 0), mode='constant')

        pred_mag = np.hypot(pred_x, pred_y)
        gt_mag = np.hypot(gt_x, gt_y)

        error_map = (pred_mag - gt_mag) ** 2
        grad_score = np.sum(error_map) / 10.0
        return grad_score


class CONN(AvgMeter):
    def compute_metric(self, pred, gt, step=0.1):
        h, w = pred.shape
        thresh_steps = np.arange(0, 1.1, step)
        l_map = -1 * np.ones((h, w), dtype=np.float32)

        for i in range(1, thresh_steps.size):
            pd_th = pred >= thresh_steps[i]
            gt_th = gt >= thresh_steps[i]
            label_image = label(pd_th & gt_th, connectivity=1)
            regions = regionprops(label_image)

            if not regions:
                continue

            max_area = max(region.area for region in regions)
            max_region = next(region for region in regions if region.area == max_area)

            omega = np.zeros((h, w), dtype=np.float32)
            coords = max_region.coords
            omega[coords[:, 0], coords[:, 1]] = 1

            unmarked = (l_map == -1) & (omega == 0)
            l_map[unmarked] = thresh_steps[i - 1]

            dist_maps = morphology.distance_transform_edt(omega == 0)
            if dist_maps.max() > 0:
                dist_maps = dist_maps / dist_maps.max()

        l_map[l_map == -1] = 1.0

        d_pd = pred - l_map
        d_gt = gt - l_map
        phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
        phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)

        loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
        return loss
