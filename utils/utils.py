import argparse
import torch
import os
import sys
import logging
from torchvision import transforms
import functools
import shutil
import torch.nn.functional as F

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def check_dir(dir):
    # use try except, instead if not path.exists
    # as it may meet files exist error when multi-GPUs
    # a_process -> os.mkdir
    # os.mkdir -> b_process
    try:
        os.makedirs(dir)
    except OSError:
        pass
    return dir

def check_txt(dir,filename):
    if os.path.exists(os.path.join(dir,filename)):
        pass
    else:
        with open(os.path.join(dir,filename), "w") as file:
            print("txt files created successfully.")



def setup_logger(name, save_dir, distributed_rank, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def reverse_normalize(normalized_image):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    inv_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    inv_tensor = inv_normalize(normalized_image)
    return inv_tensor


def Tensor2PIL(img_tensor):
    func = transforms.ToPILImage()
    return func(img_tensor)


def copy_folder_without_images(src_folder, dest_folder, image_extensions=('.jpg', '.jpeg', '.png', '.gif')):
    # copy the folder name for save results, avoiding process preemption in acceleration when mkdir
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Walk through the source folder
    for root, dirs, files in os.walk(src_folder):
        # Exclude images with specified extensions
        files = [file for file in files if not file.endswith(image_extensions)]

        # Create the corresponding sub-directory structure in the destination folder
        dest_root = os.path.join(dest_folder, os.path.relpath(root, src_folder))
        if not os.path.exists(dest_root):
            os.makedirs(dest_root)

        # Copy non-image files to the destination
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)
            shutil.copy2(src_file, dest_file)  # Use shutil.copy2 to preserve file metadata


def compute_tensor_iu(seg, gt):
    intersection = (seg & gt).float().sum()
    union = (seg | gt).float().sum()

    return intersection, union


def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou



