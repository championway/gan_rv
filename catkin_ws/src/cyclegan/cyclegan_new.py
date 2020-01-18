import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets_box import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import pytorch_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="sim2real_cycle_ssim", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=240, help='size of image height')
parser.add_argument('--img_width', type=int, default=320, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=8525, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()
print(opt)

save_root = '/media/arg_ws3/5E703E3A703E18EB/research/cycle_box_ssim/'
data_root = "/media/arg_ws3/5E703E3A703E18EB/data/mm_unity_new/"
root_A_train = "/media/arg_ws3/5E703E3A703E18EB/data/mm_unity_new/unity_boxes_train.csv"
root_A_test = "/media/arg_ws3/5E703E3A703E18EB/data/mm_unity_new/unity_boxes_test.csv"
root_B_train = "/media/arg_ws3/5E703E3A703E18EB/data/mm_unity_new/real_boxes_train.csv"
root_B_test = "/media/arg_ws3/5E703E3A703E18EB/data/mm_unity_new/real_boxes_test.csv"

# Create sample and checkpoint directories
os.makedirs(save_root + 'images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs(save_root + 'saved_models/%s' % opt.dataset_name, exist_ok=True)

class SimilarMSELoss(nn.Module):
    def __init__(self):
        super(SimilarMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        #pred_mean = pred.detach()
        #target_mean = target.detach()
        #for i in range(pred.size()[0]):
        #    pred[i] = pred[i] - pred[i].mean()
        #    target[i] = target[i] - target.mean()
        diff = target.mean() - pred.mean()
        self.loss = (diff ** 2).mean()
        return self.loss

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
D_A_identity = torch.nn.L1Loss()
D_B_identity = torch.nn.L1Loss()
criterion_similar = SimilarMSELoss()
ssim_loss = pytorch_ssim.SSIM()

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
    D_A_identity.cuda()
    D_B_identity.cuda()
    criterion_similar.cuda()
    ssim_loss.cuda()

pretrained_num = 0

if False:
    # Load pretrained models
    G_AB.load_state_dict(torch.load(save_root + 'saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, pretrained_num)))
    G_BA.load_state_dict(torch.load(save_root + 'saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, pretrained_num)))
    D_A.load_state_dict(torch.load(save_root + 'saved_models/%s/D_A_%d.pth' % (opt.dataset_name, pretrained_num)))
    D_B.load_state_dict(torch.load(save_root + 'saved_models/%s/D_B_%d.pth' % (opt.dataset_name, pretrained_num)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Loss weights
lambda_cyc = 10
# lambda_id = 3
lambda_similarity = 1
# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

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

# Training data loader
dataloader = DataLoader(ImageDataset(data_root, root_A_train, root_B_train, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
# Test data loader
val_dataloader = DataLoader(ImageDataset(data_root, root_A_test, root_B_test, transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=5, shuffle=True, num_workers=1)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs['B'].type(Tensor))
    fake_A = G_BA(real_B)
    img_sample = torch.cat((real_A.data, fake_B.data,
                            real_B.data, fake_A.data), 0)
    save_image(img_sample, save_root + 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(batch['A'].type(Tensor))  # Virtual
        real_B = Variable(batch['B'].type(Tensor))  # REAL

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        fake_B = G_AB(real_A)
        recov_A = G_BA(fake_B)
        fake_A = G_BA(real_B)
        recov_B = G_AB(fake_A)

        # Identity loss
        # loss_id_A = criterion_identity(G_BA(real_A), real_A)
        # loss_id_B = criterion_identity(G_AB(real_B), real_B)
        # # loss_id_A = criterion_identity(real_A, fake_B)
        # # loss_id_B = criterion_identity(real_B, fake_A)
        # loss_identity = (loss_id_A + loss_id_B) / 2

        # SSIM loss
        loss_ssim_A = 1 - ssim_loss(fake_B, real_A)
        loss_ssim_B = 1 - ssim_loss(fake_A, real_B)
        # loss_ssim_A = 1 - ssim_loss(G_BA(real_A), real_A)
        # loss_ssim_B = 1 - ssim_loss(G_AB(real_B), real_B)

        loss_similarity = (loss_ssim_A + loss_ssim_B) / 2

        # Cycle loss
        loss_cycle_A = criterion_cycle(real_A, recov_A)
        loss_cycle_B = criterion_cycle(real_B, recov_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # GAN loss
        loss_GAN_A = criterion_GAN(D_A(fake_A), valid)
        loss_GAN_B = criterion_GAN(D_B(fake_B), valid)

        loss_GAN = (loss_GAN_A + loss_GAN_B) / 2

        # loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity 
        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_similarity * loss_similarity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_valid = criterion_GAN(D_A(real_A), valid)
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_valid + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # Fake loss (on batch of previously generated samples)
        # fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        # loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Identity loss
        #loss_id_D_A = D_A_identity(G_BA(real_A), real_A)
        # Total loss
        # loss_D_A = (loss_real + loss_fake) / 2

        # loss_D_A.backward()
        # optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        loss_valid = criterion_GAN(D_B(real_B), valid)
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = (loss_valid + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        # # Real loss
        # loss_real = criterion_GAN(D_B(real_B), valid)
        # # Fake loss (on batch of previously generated samples)
        # fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        # loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # # Identity loss
        # #loss_id_D_B = D_B_identity(G_AB(real_A), real_B)
        # # Total loss
        # loss_D_B = (loss_real + loss_fake) / 2

        # loss_D_B.backward()
        # optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_cycle.item(), time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done+pretrained_num)
        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), save_root + 'saved_models/%s/G_AB_%d_.pth' % (opt.dataset_name, batches_done+pretrained_num))
            torch.save(G_BA.state_dict(), save_root + 'saved_models/%s/G_BA_%d_.pth' % (opt.dataset_name, batches_done+pretrained_num))
            torch.save(D_A.state_dict(), save_root + 'saved_models/%s/D_A_%d_.pth' % (opt.dataset_name, batches_done+pretrained_num))
            torch.save(D_B.state_dict(), save_root + 'saved_models/%s/D_B_%d_.pth' % (opt.dataset_name, batches_done+pretrained_num))


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

torch.save(G_AB.state_dict(), save_root + 'saved_models/%s/G_AB.pth' % (opt.dataset_name))
torch.save(G_BA.state_dict(), save_root + 'saved_models/%s/G_BA.pth' % (opt.dataset_name))
torch.save(D_A.state_dict(), save_root + 'saved_models/%s/D_A.pth' % (opt.dataset_name))
torch.save(D_B.state_dict(), save_root + 'saved_models/%s/D_B.pth' % (opt.dataset_name))
