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
        obj_exist = 0
        while(not obj_exist):
            image = cv2.imread(os.path.join(self.rgb_dir, self.img_list[idx]),cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(os.path.join(self.label_dir, self.img_list[idx]), cv2.IMREAD_GRAYSCALE)
            h, w = image.shape[:2]
            objs = self.bbx(image, mask)
            obj_exist = len(objs)
            idx = np.random.randint(0, len(self.img_list))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_list = []

        for obj in objs:
            region = [int(obj[1] - self.border), int(obj[1] + obj[3] + self.border),\
                  int(obj[0] - self.border), int(obj[0] + obj[2] + self.border)]

            bbx_image = image[region[0] : region[1], region[2]: region[3]]
            bbx_mask = mask[region[0] : region[1], region[2]: region[3]]
            crop_list.append([bbx_image, bbx_mask])
        crop_idx = np.random.randint(0, len(crop_list))
        bbx_image = crop_list[crop_idx][0]
        bbx_mask = crop_list[crop_idx][1]

        if np.random.random() < 0.5:
            bbx_image = np.array(bbx_image)[::-1, :]
            bbx_mask = np.array(bbx_mask)[::-1, :]

        bbx_image = cv2.resize(bbx_image, (256, 256))
        bbx_mask = cv2.resize(bbx_mask, (256, 256))

        # 1 channel to 2 channel classifier (one hot encoder)
        n_class = 2
        h, w = bbx_mask.shape
        target = torch.zeros(n_class, h, w)
        bbx_mask[bbx_mask!=0] = 1 # 255 to 1
        bbx_mask = torch.from_numpy(bbx_mask.copy()).long()
        for i in range(n_class):
            target[i][bbx_mask == i] = 1

        #target = target.unsqueeze(dim=0)
        # img_depth = torch.tensor(np.array(img_depth)).unsqueeze(dim=0)
        bbx_image = Image.fromarray(bbx_image).resize((256, 256), Image.ANTIALIAS)
        bbx_image = self.transform_A(bbx_image)

        return {'A': target, 'B': bbx_image}

    def bbx(self, cv_image, label_img):
        ret, threshed_img = cv2.threshold(label_img, 0, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bbx_list = []
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            if w < 20 or h < 20:
                continue
            rnd_x = np.random.randint(0, self.rnd_border)
            rnd_y = np.random.randint(0, self.rnd_border)
            img_h, img_w = threshed_img.shape
            x_ = x - rnd_x
            w_ = w + rnd_x*2
            y_ = y - rnd_y
            h_ = h + rnd_y*2
            if x_ > 0 and y_ > 0 and x_ + w_ < img_w and y_ + h_ < img_h:
                x_crop, y_crop, w_crop, h_crop = x_, y_, w_, h_
            else:
                x_crop, y_crop, w_crop, h_crop = x, y, w, h
            tmp = label_img[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
            l = tmp[tmp!=0]
            counts = np.bincount(l)
            cls = np.argmax(counts)
            bbx_list.append([x_crop, y_crop, w_crop, h_crop, cls])
        return bbx_list

    def __len__(self):
        return len(self.img_list)
