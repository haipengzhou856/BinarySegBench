import os
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset
import json
from PIL import Image
import numpy as np
from torchvision.transforms import InterpolationMode
from .augmentations import get_train_joint_transform, get_test_joint_transform

Image.MAX_IMAGE_PIXELS = None
MAT_INTERPOLATION = InterpolationMode.BILINEAR
SEG_INTERPOLATION = InterpolationMode.NEAREST


class SampleDataset(Dataset):
    def __init__(self, sample_path):


        self.image_dir = sample_path
        self.image_filenames = os.listdir(self.image_dir)
        self.transform = transforms.Compose([
                transforms.Resize((1022, 1022)), # keep original size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # avoid add prompt here as transform will do some deformation
        # just online retrieve the box or point 
        image_name = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        w, h = image.size
        ori_hw = (h, w)
        aug_image = self.transform(image)

        return {
            "image": aug_image,
            "ori_hw": ori_hw,
            "img_path": image_path,
        }

class SampleRefRWDataset(Dataset):
    def __init__(self, dataset_path,json_path):


        self.image_dir = dataset_path
        with open(json_path, 'r', encoding='utf-8') as file:
            self.data_dict = json.load(file)
        self.data_list = []
        for file_name in self.data_dict:
            labels = self.data_dict[file_name]['labels']
            for label in labels:
                self.data_list.append((file_name, label))

        self.joint_transform = get_test_joint_transform(scale=(1022,1022))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_name, selected_label = self.data_list[index]
        data = self.data_dict[file_name]
        labels = data['labels']

        mask_path = os.path.join(self.image_dir,  "mask", selected_label + ".png")
        #random_text = random.choice(labels[selected_label]["expressions"]) # TODO

        img_path = os.path.join(self.image_dir,   "img", file_name + ".jpg")
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        w, h = image.size
        ori_hw = (h, w)

        aug_image, aug_mask = self.joint_transform(image, mask)

        return {
            "image": aug_image,
            "label": aug_mask,
            "ori_hw": ori_hw,
            "mask_path": mask_path,
            "img_path": img_path,
            "file_name": file_name,
            #"label_key": selected_label,
        }
