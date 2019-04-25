import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import cv2
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="monet2photo", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=480, help='size of image height')
parser.add_argument('--img_width', type=int, default=640, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# Initialize generator and discriminator
G_AB = GeneratorResNet(res_blocks=opt.n_residual_blocks)
G_BA = GeneratorResNet(res_blocks=opt.n_residual_blocks)
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if True:
    # opt.epoch
    # Load pretrained models
    pth_name = 18200
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, 18200)))
    G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, 18200)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.dataset_name, 18200)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.dataset_name, 18200)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

data_transform = transforms.Compose(
                [ transforms.Resize(int(opt.img_height), Image.BICUBIC),
                #transforms.RandomCrop((opt.img_height, opt.img_width)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    )

# Training data loader
'''dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)'''
# Test data loader
'''val_dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=1, shuffle=True, num_workers=1)'''


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    #imgs = next(iter(val_dataloader))
    image = cv2.imread("0_original.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(image)
    pil_im = data_transform(pil_im)
    pil_im = pil_im.unsqueeze(0)

    image = cv2.resize(image, (256, 256)) 
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image)
    image = image.unsqueeze(0)

    print(image.shape)
    print(pil_im.shape)
    #print(imgs['A'].shape)

    my_img = Variable(pil_im.type(Tensor))
    my_img_fake = G_BA(my_img)
    my_img_fake = my_img_fake.squeeze(0).detach().cpu()

    #my_img_fake = transforms.functional.to_pil_image(my_img_fake)
    #my_img_fake = transforms.ToPILImage()(my_img_fake)
    #my_img_fake = transforms.Normalize((-0.5, -0.5 -0.5), (1/0.5, 1/0.5, 1/0.5))(my_img_fake)
    #my_img_fake.show()
    #my_img_fake = transforms.Resize((480, 480), Image.BICUBIC)(my_img_fake)
    #my_img_fake = transforms.ToTensor()(my_img_fake)
    #my_img_fake = transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))(my_img_fake)
    #my_img_fake = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(my_img_fake)

    #real_A = Variable(imgs['A'].type(Tensor))
    #fake_B = G_AB(real_A)
    #real_B = Variable(imgs['B'].type(Tensor))
    #fake_A = G_BA(real_B)
    #img_sample = torch.cat((real_A.data, fake_B.data,
    #                        real_B.data, fake_A.data), 0)
    #img_sample = torch.cat((real_A.data, fake_B.data), 0)
    #img_sample = torch.cat((my_img.data, my_img_fake.data), 0)
    #save_image(img_sample, '%s.png' % (batches_done), nrow=5, normalize=True)
    '''mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    normalize = transforms.Normalize(mean.tolist(), std.tolist())
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    my_img_fake = unnormalize(my_img_fake)'''

    save_image(my_img_fake, 'boxxx.png', nrow=5, normalize=True)

sample_images(1)