import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter

from utils.metrix_v1 import compute_iou, compute_boundary_acc

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

class IoU(AvgMeter):
    def compute_metric(self, pred, gt):
        pred_norm = pred.astype(np.float32) / 255.0 if pred.max() > 1.0 else pred.astype(np.float32)
        gt_norm = gt.astype(np.float32) / 255.0 if gt.max() > 1.0 else gt.astype(np.float32)

        pred_bin = pred_norm > 0.5
        gt_bin = gt_norm > 0.5

        return compute_iou(gt_bin, pred_bin)
    
class mBA(AvgMeter):
    def compute_metric(self, pred, gt):
        pred_norm = pred.astype(np.float32) / 255.0 if pred.max() > 1.0 else pred.astype(np.float32)
        gt_norm = gt.astype(np.float32) / 255.0 if gt.max() > 1.0 else gt.astype(np.float32)

        pred_bin = pred_norm > 0.5
        gt_bin = gt_norm > 0.5

        return compute_boundary_acc(gt_bin, pred_bin)
