# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

### Borrow from https://github.com/SysCV/sam-hq/blob/main/train/utils/misc.py
### thanx for code sharing

import os
import random
import subprocess
import time
from collections import OrderedDict, defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import json, time
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle




## note I chaged the threshold as in HQSAM their loader read [0~255] instead of [0~1]
## GPT improved version :-/
def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    batch_size, height, width = masks.shape
    boxes = torch.zeros((batch_size, 4), dtype=torch.int64, device=masks.device)

    '''if masks.device.type == 'cuda':
        y_indices = torch.arange(height, device=masks.device).view(1, height, 1)
        x_indices = torch.arange(width, device=masks.device).view(1, 1, width)

        row_valid = masks.any(dim=2)  # [N, H]
        col_valid = masks.any(dim=1)  # [N, W]

        y_min = row_valid.float().cumsum(dim=1).eq(1).argmax(dim=1)
        y_max = (row_valid.flip(dims=[1]).float().cumsum(dim=1).eq(1).argmax(dim=1)).flip(dims=[0])
        x_min = col_valid.float().cumsum(dim=1).eq(1).argmax(dim=1)
        x_max = (col_valid.flip(dims=[1]).float().cumsum(dim=1).eq(1).argmax(dim=1)).flip(dims=[0])

        return torch.stack([x_min, y_min, x_max, y_max], dim=1)'''

    for i in range(batch_size):
        mask = masks[i] > 0.5  #
        if not mask.any():
            continue

        y = torch.where(mask.any(dim=1))[0]
        x = torch.where(mask.any(dim=0))[0]
        boxes[i] = torch.tensor([x[0], y[0], x[-1], y[-1]], dtype=torch.int64)

    return boxes

def masks_sample_points(masks: torch.Tensor, k: int = 10) -> torch.Tensor:
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    batch_size, height, width = masks.shape
    points = torch.zeros((batch_size, k, 2), dtype=torch.float32, device=masks.device)

    masks_bin = masks > 0.5 # 0-1 threshold

    '''if masks.device.type == 'cuda' and batch_size > 1:
        num_points = masks_bin.sum(dim=(1, 2))  # [N]
        valid_mask = num_points > 0
        y_idx, x_idx = torch.where(masks_bin[valid_mask])
        batch_idx = torch.arange(valid_mask.sum(), device=masks.device)
        ptr = 0
        for i in range(batch_size):
            if not valid_mask[i]:
                continue
            cnt = num_points[i].item()
            sample_cnt = min(k, cnt)
            perm = torch.randperm(cnt, device=masks.device)[:sample_cnt]
            batch_points = torch.stack([
                x_idx[ptr + perm].float(),
                y_idx[ptr + perm].float()
            ], dim=1)
            points[i, :sample_cnt] = batch_points
            ptr += cnt
        return points'''

    for i in range(batch_size):
        mask = masks_bin[i]
        if not mask.any():
            continue

        y, x = torch.where(mask)
        cnt = y.size(0)
        sample_cnt = min(k, cnt)

        if sample_cnt < k:
            pad = torch.zeros((k - sample_cnt, 2), dtype=torch.float32, device=masks.device)
            points[i] = torch.cat([torch.stack([x[:sample_cnt], y[:sample_cnt]], dim=1), pad], dim=0)
            continue

        perm = torch.randperm(cnt, device=masks.device)[:k]
        points[i] = torch.stack([x[perm], y[perm]], dim=1).float()

    return points


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def show_pos_points(coords, ax, marker_size=375):
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_neg_points(coords, ax, marker_size=375):
    ax.scatter(coords[:, 0], coords[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def visualize_with_box_and_points(labels, boxes, points,idx=0):
    fig = plt.figure()


    show_mask(labels[idx].cpu(), plt.gca())
    show_box(boxes[idx].cpu(), plt.gca())
    show_pos_points(points[idx].cpu(), plt.gca())


    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    data = torch.from_numpy(data).permute(2, 0, 1).float() / 255.0
    return data