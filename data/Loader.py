from .ImgSegDataset import IMGDataset
from torch.utils.data import DataLoader


def GetIMGLoader(config, is_train=True):
    batch_size = config.DATALOADER.BATCH_SIZE
    num_workers = config.DATALOADER.NUM_WORKERS

    if is_train:
        img_dataset = IMGDataset(config, is_train=is_train)
        img_dataloader = DataLoader(dataset=img_dataset,
                                    batch_size=batch_size,drop_last=False,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers, )
    else:
        img_dataset = IMGDataset(config, is_train=is_train)
        img_dataloader = DataLoader(dataset=img_dataset,
                                    batch_size=batch_size, # always keep bs=1 in testing stage
                                    shuffle=False,
                                    num_workers=num_workers, )
    return img_dataloader


def GetRefLoader(config, is_train=True):
    batch_size = config.DATALOADER.BATCH_SIZE
    num_workers = config.DATALOADER.NUM_WORKERS

    if is_train:
        img_dataset = RefMatDataset(config, is_train=is_train)
        img_dataloader = DataLoader(dataset=img_dataset,
                                    batch_size=batch_size,drop_last=False,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers, )
    else:
        img_dataset = RefMatDataset(config, is_train=is_train)
        img_dataloader = DataLoader(dataset=img_dataset,
                                    batch_size=batch_size, # always keep bs=1 in testing stage
                                    shuffle=False,
                                    num_workers=num_workers, )
    return img_dataloader