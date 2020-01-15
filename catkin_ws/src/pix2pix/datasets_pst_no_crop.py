import glob
import random
import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

class ImageDataset(Dataset):
    def __init__(self, root = "/media/arg_ws3/5E703E3A703E18EB/data/PST900_RGBT_Dataset/", transforms_A=None, transforms_B=None, mode='train'):
        self.transform_A = transforms.Compose(transforms_A)
        self.transform_B = transforms.Compose(transforms_B)
        self.mode = mode
        self.root = root
        self.border = 0
        self.rnd_border = 20
        self.files = []
        data_dir = os.path.join(root, mode)
        self.img_list = os.listdir(os.path.join(data_dir, 'rgb'))
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.label_dir = os.path.join(data_dir, 'labels')

    def __getitem__(self, index):
        idx = index % len(self.img_list)
        image = cv2.imread(os.path.join(self.rgb_dir, self.img_list[idx]),cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(os.path.join(self.label_dir, self.img_list[idx]), cv2.IMREAD_GRAYSCALE)
        h, w = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_list = []

        if np.random.random() < 0.5 and self.mode == 'train':
            image = np.array(image)[:-1:, :]
            mask = np.array(mask)[:-1:, :]

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # 1 channel to 2 channel classifier (one hot encoder)
        n_class = 5
        h, w = mask.shape
        target = torch.zeros(n_class, h, w)
        # mask[mask!=0] = 1 # 255 to 1
        mask_ = torch.from_numpy(mask.copy()).long()
        for i in range(n_class):
            target[i][mask_ == i] = 1

        #target = target.unsqueeze(dim=0)
        # img_depth = torch.tensor(np.array(img_depth)).unsqueeze(dim=0)
        image = Image.fromarray(image).resize((256, 256), Image.ANTIALIAS)
        image = self.transform_A(image)

        return {'A': target, 'B': image, 'l':mask}

    def __len__(self):
        return len(self.img_list)
