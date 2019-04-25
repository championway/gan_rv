import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.root = root

        self.files_A = []
        self.files_B = []

        for line in open(os.path.join(root, 'main', 'unity_' + mode + '.txt')):
            self.files_A.append(line.strip())
        for line in open(os.path.join(root, 'main', 'real_' + mode + '.txt')):
            self.files_B.append(line.strip())

    def __getitem__(self, index):
        A_name = self.files_A[random.randint(0, len(self.files_A) - 1)]
        img_A = Image.open(self.root + 'unity_boxes/' + A_name)
        item_A = self.transform(img_A)

        B_name = self.files_B[index % len(self.files_B)]
        img_B = Image.open(self.root + 'real_boxes/' + B_name)
        item_B = self.transform(img_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
