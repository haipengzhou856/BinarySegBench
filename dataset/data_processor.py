import os
from torch.utils.data import Dataset
from .augmentations import get_train_joint_transform, get_test_joint_transform
from torch.utils.data import DataLoader
from PIL import Image


## This CustomDataset is compatible with a dataset by follow structure:
## --XXDataset
## ----Train
## ----Test
## ------img
## --------xx.jpg
## ------mask
## --------xx.png

## Modify it with different settings

class CustomDataset(Dataset):
    def __init__(self, config, is_train):
        if is_train:
            self.mode = "train"
        else:
            self.mode = "test"
        self.data_dir = os.path.join(config.DATASET.DATA_ROOT, self.mode)

        self.image_dir = os.path.join(self.data_dir, "img")
        self.mask_dir = os.path.join(self.data_dir, "mask")
        self.image_filenames = os.listdir(self.image_dir)

        if self.mode == "train":
            self.joint_transform = get_train_joint_transform(scale=(config.DATASET.IMG_SIZE,
                                                                    config.DATASET.IMG_SIZE))
        else:
            self.joint_transform = get_test_joint_transform(scale=(config.DATASET.IMG_SIZE,
                                                                   config.DATASET.IMG_SIZE))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_name = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.split(".")[0]+".png")

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        w, h = image.size

        aug_image, aug_mask = self.joint_transform(image, mask)

        return {"image": aug_image, "label": aug_mask,
                "ori_w": w, "ori_h": h, "mask_path": mask_path}


def GetDataLoader(config, is_train=True):
    batch_size = config.DATASET.BATCH_SIZE
    num_workers = config.DATASET.NUM_WORKERS

    if is_train:
        seg_dataset = CustomDataset(config, is_train=is_train)
        seg_dataloader = DataLoader(dataset=seg_dataset,
                                    batch_size=batch_size,drop_last=False,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers)
    else:
        seg_dataset = CustomDataset(config, is_train=is_train)
        seg_dataloader = DataLoader(dataset=seg_dataset,
                                    batch_size=1, # always keep bs=1 in testing stage
                                    shuffle=False,
                                    num_workers=num_workers)
    return seg_dataloader
