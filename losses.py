#from https://github.com/ajbrock/BigGAN-PyTorch (MIT license) - no modifications
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss

def discriminator_loss_hinge_fake(dis_fake, weight_fake = None):
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_fake

def discriminator_loss_hinge_real(dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    return loss_real

def loss_hinge_gen(dis_fake, weight_fake = None):
  loss = -torch.mean(dis_fake)
  return loss


def loss_hinge_dis(dis_fake, dis_real, weight_real = None, weight_fake = None):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
