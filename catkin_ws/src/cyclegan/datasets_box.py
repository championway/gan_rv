import glob
import random
import os
import csv
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, root_A, root_B, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.root = root
        
        self.data_A = pd.read_csv(root_A)
        self.data_B = pd.read_csv(root_B)

    def __getitem__(self, index):
        A_name = self.data_A.iloc[random.randint(0, len(self.data_A) - 1), 0]
        img_A = Image.open(os.path.join(self.root + A_name))
        item_A = self.transform(img_A)

        B_name = self.data_B.iloc[index % len(self.data_B), 0]
        img_B = Image.open(os.path.join(self.root + B_name))
        item_B = self.transform(img_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.data_A), len(self.data_B))
