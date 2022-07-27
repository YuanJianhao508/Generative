import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class WaterBirdsDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None, intervention=False, datadir="metadata.csv"):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        metadata_df = pd.read_csv(os.path.join(basedir, datadir))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        # print(len(self.metadata_df))
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['y'].values
        self.p_array = self.metadata_df['place'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['place'].values
        self.n_places = np.unique(self.confounder_array).size
        self.n_groups = self.n_classes * self.n_places
        self.filename_array = self.metadata_df['img_filename'].values

        # For logging
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()

        self.intervention = intervention

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        # y label: class label 0-landbird (stand) 1-waterbird (flying)
        # p background: place label 0-land 1-water
        y = self.y_array[idx]
        p = self.confounder_array[idx]
        g = self.group_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        # img.show()

        if self.transform:
            img = self.transform(img)

        #Intervention Stage?
        if self.intervention:
            return img, y, p, self.filename_array[idx]
        else:
            return img, y, g, p

    def get_target_mapping(self,dataset):
        if dataset == 'waterbird':
            return {0:'bird',1:'bird'}
        elif dataset == 'celebA':
            return {0:'non-blond',1:'blond'}
def get_transform_cub(target_resolution, train, augment_data):
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_loader(data, train, **kwargs):
    if train: # Validation or testing
        shuffle = True
    else:
        shuffle = False
    loader = DataLoader(
        data,
        shuffle=shuffle,
        sampler=None,
        **kwargs)
    return loader


def log_data(logger, train_data, test_data, val_data=None, get_yp_func=None):
    logger.write(f'Training Data (total {len(train_data)})\n')
    for group_idx in range(train_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {train_data.group_counts[group_idx]:.0f}\n')
    logger.write(f'Test Data (total {len(test_data)})\n')
    for group_idx in range(test_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {test_data.group_counts[group_idx]:.0f}\n')
    if val_data is not None:
        logger.write(f'Validation Data (total {len(val_data)})\n')
        for group_idx in range(val_data.n_groups):
            y_idx, p_idx = get_yp_func(group_idx)
            logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {val_data.group_counts[group_idx]:.0f}\n')
