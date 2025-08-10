import os
import torch
from torch.utils.data import Dataset
from .augmentations import get_train_joint_transform, get_test_joint_transform
from PIL import Image
import numpy as np
from torchvision.transforms import InterpolationMode

Image.MAX_IMAGE_PIXELS = None
MAT_INTERPOLATION = InterpolationMode.BILINEAR
SEG_INTERPOLATION = InterpolationMode.NEAREST


class IMGDataset(Dataset):
    def __init__(self, config, is_train):
        if is_train:
            self.mode = "train"
        else:
            self.mode = "test"
        self.data_dir = os.path.join(config.DATASET.ROOT, self.mode)

        self.image_dir = os.path.join(self.data_dir, "im")
        self.mask_dir = os.path.join(self.data_dir, "gt")
        self.image_filenames = sorted(os.listdir(self.image_dir))

        self.img_path_to_idx = {}
        self.mask_path_to_idx = {}
        self.idx_to_img_path = {}  # Changed from list to dictionary
        self.idx_to_mask_path = {}  # Changed from list to dictionary

        for idx, filename in enumerate(self.image_filenames):
            img_path = os.path.join(self.image_dir, filename)
            mask_filename = f"{os.path.splitext(filename)[0]}.png"
            mask_path = os.path.join(self.mask_dir, mask_filename)

            self.img_path_to_idx[img_path] = idx
            self.mask_path_to_idx[mask_path] = idx

            self.idx_to_img_path[idx] = img_path  # Store in dictionary with index as key
            self.idx_to_mask_path[idx] = mask_path  # Store in dictionary with index as key

        if self.mode == "train":
            self.joint_transform = get_train_joint_transform(scale=(config.INPUT.SIZE,
                                                                    config.INPUT.SIZE)
                                                             )
        else:
            self.joint_transform = get_test_joint_transform(scale=(config.INPUT.SIZE,
                                                                   config.INPUT.SIZE)
                                                            )

    def __len__(self):
        return len(self.image_filenames)

    def get_img_path(self, idx):
        # Check if index exists in dictionary
        if idx in self.idx_to_img_path:
            return self.idx_to_img_path[idx]
        raise IndexError(f"Index {idx} out of range")

    def get_mask_path(self, idx):
        # Check if index exists in dictionary
        if idx in self.idx_to_mask_path:
            return self.idx_to_mask_path[idx]
        raise IndexError(f"Index {idx} out of range")

    def get_idx_by_img_path(self, img_path):
        return self.img_path_to_idx.get(img_path, -1)

    def get_idx_by_mask_path(self, mask_path):
        return self.mask_path_to_idx.get(mask_path, -1)

    def __getitem__(self, index):
        # Check if index is valid before proceeding
        if index not in self.idx_to_img_path or index not in self.idx_to_mask_path:
            raise IndexError(f"Index {index} is invalid")

        image_path = self.get_img_path(index)
        mask_path = self.get_mask_path(index)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        w, h = image.size
        ori_hw = (h, w)

        aug_image, aug_mask = self.joint_transform(image, mask)

        return {
            "image": aug_image,
            "label": aug_mask,
            "ori_hw": ori_hw,
            "mask_idx": index,
            "img_idx": index,
        }
