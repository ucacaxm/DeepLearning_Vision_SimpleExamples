# -*- coding: utf-8 -*-
"""
Corrigé : DCGAN simple et "beau" sur MNIST / FashionMNIST
Téléchargeable & exécutable dans Colab / local Python.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# -------------------------------
# Configuration / hyperparamètres
# -------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

dataset_name = "MNIST"  # ou "FashionMNIST"
data_root = "./data"
out_dir = "./outputs_gan"
os.makedirs(out_dir, exist_ok=True)

image_size = 28
nc = 1                # channels (1 pour MNIST)
nz = 100              # dimension du bruit latent
ngf = 64              # feature maps générateur
ndf = 64              # feature maps discriminateur
batch_size = 128
lr = 2e-4
beta1 = 0.5
beta2 = 0.999
epochs = 30
save_every = 5        # sauvegarde / génération d'images tous les N epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Transforms & DataLoader
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # images dans [-1,1]
])

if dataset_name == "MNIST":
    dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
else:
    dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# -------------------------------
# Utilitaires
# -------------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# -------------------------------
# Générateur (DCGAN-like)
# -------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture adaptée à 28x28 (MNIST)
        # On part d'un vecteur latent z (nz x 1 x 1) et on applique une série de convtranspose
        self.net = nn.Sequential(
            # input: Nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size: (ngf*4) x 3 x 3
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2) x 6 x 6
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf) x 12 x 12
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # output: nc x 28 x 28 (approx, architecture choisie pour arriver à 28)
        )

    def forward(self, z):
        return self.net(z)

# -------------------------------
# Discriminateur (DCGAN-like)
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # input nc x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 14 x 14
