# -*- coding: utf-8 -*-
"""
DCGAN complet pour MNIST / FashionMNIST
Entraînement + visualisation + génération finale
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
nc = 1
nz = 100
ngf = 64    # taille feature maps générateur
ndf = 64    # taille feature maps discriminateur
batch_size = 128
lr = 2e-4
beta1 = 0.5
beta2 = 0.999
epochs = 30
save_every = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Transforms & DataLoader
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if dataset_name == "MNIST":
    dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
else:
    dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# -------------------------------
# Utils
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
# Générateur (DCGAN)
# -------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# -------------------------------
# Discriminateur (DCGAN)
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

# -------------------------------
# Initialisation
# -------------------------------
G = Generator().to(device)
D = Discriminator().to(device)

G.apply(weights_init_normal)
D.apply(weights_init_normal)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

# -------------------------------
# Entraînement
# -------------------------------
print("Début de l'entraînement...")

for epoch in range(1, epochs + 1):
    for imgs, _ in loader:

        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # ----- Train D -----
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_imgs = G(z).detach()

        D_real = D(real_imgs)
        loss_real = criterion(D_real, real_labels)

        D_fake = D(fake_imgs)
        loss_fake = criterion(D_fake, fake_labels)

        loss_D = loss_real + loss_fake

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ----- Train G -----
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_imgs = G(z)

        D_fake = D(fake_imgs)
        loss_G = criterion(D_fake, real_labels)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"[Epoch {epoch}/{epochs}]  Loss_D={loss_D.item():.4f} | Loss_G={loss_G.item():.4f}")

    # ----- Sauvegarde intermédiaire -----
    if epoch % save_every == 0:
        with torch.no_grad():
            fake = G(fixed_noise).cpu()
        grid = utils.make_grid(fake, nrow=8, normalize=True)
        plt.figure(figsize=(6,6))
        plt.imshow(grid.permute(1,2,0))
        plt.axis("off")
        plt.savefig(f"{out_dir}/epoch_{epoch}.png")
        plt.close()

# -------------------------------
# Génération finale
# -------------------------------
print("Génération finale...")
with torch.no_grad():
    fake = G(fixed_noise).cpu()

grid = utils.make_grid(fake, nrow=8, normalize=True)
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1,2,0))
plt.axis("off")
plt.show()
