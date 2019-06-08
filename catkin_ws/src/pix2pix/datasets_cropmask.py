import glob
import random
import os
import numpy as np

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
    def __init__(self, root = "/media/arg_ws3/5E703E3A703E18EB/data/subt_all/", transforms_A=None, transforms_B=None, mode='train'):
        self.transform_A = transforms.Compose(transforms_A)
        self.transform_B = transforms.Compose(transforms_B)
        self.root = root
        self.files = []
        self.obj_class = ['bb_extinguisher', 'bb_drill', 'bb_backpack']
        for line in open(os.path.join(root, mode + '.txt')):
            self.files.append(line.strip())

    def __getitem__(self, index):
        idx = index % len(self.files)
        ann_path = self.root + 'Annotations/' + self.files[idx] + '.xml'
        mask_path = self.root + 'mask/' + self.files[idx] + '.png'
        depth_path = self.root + 'depth/' + self.files[idx] + '.png'
        rgb_path = self.root + 'image/' + self.files[idx] + '.jpg'

        bbx = self.get_ann(ann_path)

        img_mask = Image.open(mask_path).crop(bbx).resize((256, 256), Image.ANTIALIAS)
        img_depth = Image.open(depth_path).crop(bbx).resize((256, 256), Image.ANTIALIAS)
        img_rgb = Image.open(rgb_path).crop(bbx).resize((256, 256), Image.ANTIALIAS)

        img_mask = img_mask.convert('L')
        # img_depth = img_depth.convert('L')

        img_mask = np.array(img_mask)
        img_depth = np.array(img_depth)/1000.
        img_rgb = np.array(img_rgb)
        # print(img_depth.max(), img_depth.min(), img_depth.shape)

        depth_max = float(img_depth.max())
        depth_min = float(img_depth.min())
        # print(depth_min, depth_max)

        img_depth = (img_depth - depth_min)/(depth_max - depth_min)

        if np.random.random() < 0.5:
            img_mask = np.array(img_mask)[::-1, :]
            img_depth = np.array(img_depth)[::-1, :]
            img_rgb = np.array(img_rgb)[::-1, :]

        # img_A = torch.tensor(np.array(img_A)).unsqueeze(dim=0)

        #img_mask = self.transform_A(Image.fromarray(img_mask, 'L'))
        img_mask = torch.tensor(np.array(img_mask)).unsqueeze(dim=0)
        img_depth = torch.tensor(np.array(img_depth)).unsqueeze(dim=0)
        #img_rgb = torch.tensor(np.array(img_rgb))
        img_rgb = self.transform_A(Image.fromarray(img_rgb))
        # print(img_mask.shape, img_depth.shape, img_rgb.shape)

        # img_A = self.transform_A(img_A)
        # img_B = self.transform_B(img_B)
        # img_C = self.transform_B(img_C)

        return {'A': img_mask, 'B': img_depth, 'C': img_rgb}

    def get_ann(self, ann_path):
        target = ET.parse(ann_path).getroot()
        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            if name not in self.obj_class:
                continue
            bbox = obj.find('bndbox')
            if bbox is not None:
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(bbox.find(pt).text) - 1
                    # scale height or width
                    #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                    bndbox.append(cur_pt)
                res += [bndbox]  # [xmin, ymin, xmax, ymax]
            else: # For LabelMe tool
                polygons = obj.find('polygon')
                x = []
                y = []
                bndbox = []
                for polygon in polygons.iter('pt'):
                    # scale height or width
                    x.append(int(polygon.find('x').text))
                    y.append(int(polygon.find('y').text))
                bndbox.append(min(x))
                bndbox.append(min(y))
                bndbox.append(max(x))
                bndbox.append(max(y))
                res += [bndbox] # [xmin, ymin, xmax, ymax]
        return res[0]

    def __len__(self):
        return len(self.files)
