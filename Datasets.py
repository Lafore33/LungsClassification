import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (Compose, ToDtype, PILToTensor, Normalize, RandomRotation, RandomHorizontalFlip,
                                       CenterCrop, Pad, Resize)
from PIL import Image
from random import randrange


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, mask_mode=0):
        data = pd.read_csv('./data/train_answers.csv')
        data['img_name'] = data['id'].apply(lambda x: f'img_{x}.png')

        self.img_labels = data
        self.train_dir = './data/train_images/'
        self.mask_dir = './data/train_lung_masks/'
        self.transform = Compose([PILToTensor(), ToDtype(torch.float32, scale=True), Resize(224, antialias=None)])
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        train_img_path = os.path.join(self.train_dir, self.img_labels.iloc[idx, 2])
        mask_img_path = os.path.join(self.mask_dir, self.img_labels.iloc[idx, 2])
        train_image = Image.open(train_img_path)
        mask_image = Image.open(mask_img_path)

        label = self.img_labels.iloc[idx, 1]
        train_image = self.transform(train_image)
        mask_image = self.transform(mask_image)
        if self.mask_mode == 0:
            image = train_image
        elif self.mask_mode == 1:
            image = train_image * mask_image
        else:
            image = train_image * 0.5 + mask_image * 0.5

        image = Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))(image) * 0.25 + 0.5
        label = torch.tensor(label, dtype=torch.long)

        return image, label


class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        data = pd.DataFrame(index=range(6920), columns=['id', 'img_name'])
        data['id'] = range(6920)
        data['img_name'] = data['id'].apply(lambda x: f'img_{x}.png')
        self.img_labels = data
        self.test_dir = './data/test_images/'
        self.transform = Compose([PILToTensor(), ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        test_img_path = os.path.join(self.test_dir, self.img_labels.iloc[idx, 1])
        test_image = Image.open(test_img_path)
        test_image = self.transform(test_image)
        image = Resize(224, antialias=None)(test_image)
        image = Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))(image) * 0.25 + 0.5
        return image


def augmenting(image):
    aug_stack = Compose([RandomRotation(degrees=10), RandomHorizontalFlip(0.8)])
    image = aug_stack(image)
    zoom_size = randrange(-10, 11)
    if zoom_size < 0:
        image = CenterCrop(int(224 * (1 + zoom_size / 100)))(image)
    else:
        image = Pad(int(224 * zoom_size / 100))(image)
    return image
