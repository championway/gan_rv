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
    def __init__(self, root = "/media/arg_ws3/5E703E3A703E18EB/data/mm_barcode/", transforms_A=None, transforms_B=None, mode='train'):
        self.transform_A = transforms.Compose(transforms_A)
        self.transform_B = transforms.Compose(transforms_B)
        self.root = root
        self.files = []
        self.HEIGHT = 480
        self.WIDTH = 640
        self.obj_class = ['barcode']
        for line in open(os.path.join(root, mode + '.txt')):
            self.files.append(line.strip())

    def __getitem__(self, index):
        idx = index % len(self.files)
        ann_path = self.root + 'Annotations/' + self.files[idx] + '.xml'
        #mask_path = self.root + 'mask/' + self.files[idx] + '.png'
        #depth_path = self.root + 'depth/' + self.files[idx] + '.png'
        rgb_path = self.root + 'Images/' + self.files[idx] + '.jpg'

        mask, bbx = self.get_ann(ann_path)

        img_mask = Image.fromarray(mask)
        img_mask = img_mask.crop(bbx).resize((256, 256), Image.ANTIALIAS)
        # img_depth = Image.open(depth_path).crop(bbx).resize((256, 256), Image.ANTIALIAS)
        img_rgb = Image.open(rgb_path).crop(bbx).resize((256, 256), Image.ANTIALIAS)
        img_mask.save("MASK.png")
        img_rgb.save("RGB.png")

        img_mask = img_mask.convert('L')

        img_mask = np.array(img_mask)

        # img_depth = np.array(img_depth)/1000.
        img_rgb = np.array(img_rgb)

        # depth_max = float(img_depth.max())
        # depth_min = float(img_depth.min())

        # img_depth = (img_depth - depth_min)/(depth_max - depth_min)

        if np.random.random() < 0.5:
            img_mask = np.array(img_mask)[::-1, :]
            # img_depth = np.array(img_depth)[::-1, :]
            img_rgb = np.array(img_rgb)[::-1, :]

        # 1 channel to 2 channel classifier (one hot encoder)
        n_class = 2
        h, w = img_mask.shape
        target = torch.zeros(n_class, h, w)
        img_mask[img_mask!=0] = 1 # 255 to 1
        img_mask = torch.from_numpy(img_mask.copy()).long()
        for i in range(n_class):
            target[i][img_mask == i] = 1

        #target = target.unsqueeze(dim=0)
        # img_depth = torch.tensor(np.array(img_depth)).unsqueeze(dim=0)
        img_rgb = self.transform_A(Image.fromarray(img_rgb))

        return {'A': target, 'B': img_rgb}

    def get_ann(self, ann_path):
        target = ET.parse(ann_path).getroot()
        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            if name not in self.obj_class:
                continue
            bbox = obj.find('bndbox')
            if bbox is not None and False:
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(bbox.find(pt).text) - 1
                    # scale height or width
                    #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                    bndbox.append(cur_pt)
                res += [bndbox]  # [xmin, ymin, xmax, ymax]
            else: # For LabelMe tool
                mask = np.zeros([self.HEIGHT, self.WIDTH], dtype = np.uint8)
                polygons = obj.find('polygon')
                x = []
                y = []
                bndbox = []
                poly_vertice = []
                for polygon in polygons.iter('pt'):
                    poly_vertice.append([int(polygon.find('x').text), int(polygon.find('y').text)])
                poly_vertice = np.array(poly_vertice, np.int32)
                cv2.fillConvexPoly(mask, poly_vertice, 128)
                x_, y_, w_, h_ = cv2.boundingRect(mask)
                return mask, [x_, y_, x_+w_, y_+h_]
        return None, None

    def __len__(self):
        return len(self.files)
