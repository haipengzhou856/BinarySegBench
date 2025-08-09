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
        self.image_filenames = os.listdir(self.image_dir)

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

    def __getitem__(self, index):
        # avoid add prompt here as transform will do some deformation
        # just online retrieve the box or point 
        image_name = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.split(".")[0] + ".png")

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # for segmentation

        w, h = image.size
        ori_hw = (h, w)
        aug_image, aug_mask = self.joint_transform(image, mask)

        return {
            "image": aug_image,
            "label": aug_mask,
            "ori_hw": ori_hw,
            "mask_path": mask_path,
            "img_path": image_path,
        }


class SynthesisIMGDataset(Dataset):
    def __init__(self, config, is_train):
        if is_train:
            self.mode = "train"
        else:
            self.mode = "test"
        self.data_dir = os.path.join(config.DATASET.ROOT, self.mode)

        self.image_dir = os.path.join(self.data_dir, "im")
        self.mask_dir = os.path.join(self.data_dir, "gt")
        self.image_filenames = os.listdir(self.image_dir)
        self.bg_dir = "/home/haipeng/Data/BG20K/train"
        self.bg_filenames = os.listdir(self.bg_dir)

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

    def __getitem__(self, index):
        image_name = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.split(".")[0] + ".png")

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # [0, 255]

        w, h = image.size
        ori_hw = (h, w)

        if self.mode == "train":
            bg_index = np.random.randint(len(self.bg_filenames))
            bg_name = self.bg_filenames[bg_index]
            bg_path = os.path.join(self.bg_dir, bg_name)
            bg = Image.open(bg_path).convert('RGB').resize((w, h), Image.BILINEAR)

            bin_mask = mask.point(lambda x: 255 if x > 128 else 0, mode='1')

            bg.paste(image, mask=bin_mask)  # This modifies bg in-place
            image = bg  # now image is composite of fg + random bg

        # Apply joint transform to both image and mask
        aug_image, aug_mask = self.joint_transform(image, mask)

        return {
            "image": aug_image,
            "label": aug_mask,
            "ori_hw": ori_hw,
            "mask_path": mask_path,
            "img_path": image_path,
        }
